"""Fill in the copy numbers of the small segments that were left out of the ILP.

Pre-processing (`spice.libs.preprocess.input`) flags small segments ('filter_seg') and leaves them out of the ILP.
`fill_removed_segments` is the entry point, called by `run_spice_bootstrap` after the ILP. It infers the copy
numbers and events at the filtered positions on every solution tree:

1. In total copy number mode (`allele_specific=False`), adds a dummy second allele (copy number and events 0), to enable
   running of fill_gaps with total copy numbers (`create_pseudo_allelespecific`).
2. Pairs each flagged position with a kept neighbour: the kept position just before or just after its run of
   consecutive flagged positions, whichever has the more similar leaf profile (`get_removed_pos_closest_neighbour`,
   `compare_leaves`).
3. Builds a state tree for every solution and flagged position (`get_unique_state_trees`): the branch states
   (parent, child, copy number, WGD) of the leaves at the flagged position and of the internal nodes at the
   neighbour, taken from the ILP solution (`multi_get_internal_states`). It also records the branch states of all
   nodes at the neighbour. Positions with the same state tree and neighbour states are solved only once
   ('state_tree_id').
4. Solves each unique state tree (`multi_fill_gaps`, `fill_internal_nodes`, `fill_gaps`; see below).
5. Merges the filled positions ('cn_model' 'focal_model') with the ILP positions ('cn_model' 'ilp_model'), adds
   the event costs including the filled positions to the edge table (`add_focal_events_to_edgesdf`) and collects the
   events at the filled positions (`get_focal_events`; combined with the ILP events by
   `add_focal_events_to_eventsdf`).

Filling a state tree
--------------------
`fill_gaps` is a small parsimony (Sankoff) algorithm over copy number pairs (allele A, allele B), from the leaves up.
The tree is rooted at the most recent common ancestor of the leaves, below a fixed diploid root: (1, 1), or (2, 0)
in total copy number mode. Leaf copy numbers are fixed; for the cost they are clipped to `focal_max_cn`.

For each edge and allele, a copy number change costs 1, unless the same edge has a change in the same direction that
is at least as large at the neighbour position: the event then continues from the neighbour and costs 0
(`new_cost`). A child can only have copy number above 0 if its parent's is above 0. Among equally costly child
states, the one closest to the parent is chosen, then the one with the smaller total copy number.

Events on each edge are counted against the parent's copy number, doubled if the edge has a WGD: gains are the
increase and deletions the decrease, per allele.

Focal event costs
-----------------
`add_focal_events_to_edgesdf` adds two columns to the edge table, both summed over the edges of each solution:
'Total_cost_focal' (ILP events plus all events at filled positions) and 'Total_cost_focal_interval', which leaves
out events at filled positions where the parent and the child both have the same copy number as the position to
their left, or both as the position to their right (`cut_focal_events_by_neighbours`). Those events extend an
interval that is already counted.

Steps 3 and 4 run in a `multiprocessing.Pool` (step 3 in groups of solutions, step 4 per state tree).
`init_unique_trees` and `init_fill_gaps` store the shared inputs as module-level globals in each worker.
"""

import logging
from multiprocessing import Pool, Manager
from typing import Optional

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from Bio.Phylo.BaseTree import Tree, Clade

from spice.libs.utils.conversion import map_to_original_positions
from spice.libs.utils.logging import initialize_logger
from spice.libs.utils.progress import ProgressBar


def create_pseudo_allelespecific(df: pd.DataFrame) -> pd.DataFrame:
    """Add a dummy second allele to a total copy number table.

    Used in total copy number mode, so that the rest of the module can treat every table as allele-specific.

    Parameters
    ----------
    df : pd.DataFrame
        Table with a single allele: 'A' (leaf profiles) or 0 (ILP node copy numbers, position mapping).

    Returns
    -------
    pd.DataFrame
        `df` followed by a copy of it for the second allele ('B' if the allele is 'A', otherwise 1), with every copy
        number column (name containing 'cn') and event column (containing 'del' or 'gain'), ignoring case, set to 0.
        The index is reset.
    """
    assert len(df['Allele'].unique()) == 1, f"Expected a single allele, found {list(df['Allele'].unique())}"
    df_new_allele = df.copy()
    cn_col = [i for i in df.columns if 'cn' in i.lower()]
    for i in cn_col:
        df_new_allele[i] = 0
    events_col = [i for i in df.columns if 'del' in i.lower()] + [i for i in df.columns if 'gain' in i.lower()]
    for i in events_col:
        df_new_allele[i] = 0
    new_allele = 'B' if df['Allele'].unique()[0] == 'A' else 1
    df_new_allele['Allele'] = new_allele
    return pd.concat([df, df_new_allele]).reset_index(drop=True)


def fill_removed_segments(total_nodes: pd.DataFrame, total_edges: pd.DataFrame, all_leaves: pd.DataFrame,
                          final_mapping: pd.DataFrame, maxcn: int, j: int, focal_maxcn: Optional[int] = None,
                          logger: Optional[logging.Logger] = None, allele_specific: bool = True
                          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Infer the copy numbers and events at the segments left out of the ILP, on every solution tree.

    Each flagged segment is paired with its most similar kept neighbour (`get_unique_state_trees`), and its internal
    node copy numbers are inferred by small parsimony (`multi_fill_gaps`). The filled segments are then merged with
    the ILP output, and the edge table gets the total costs including them (`add_focal_events_to_edgesdf`).

    Parameters
    ----------
    total_nodes : pd.DataFrame
        ILP node copy numbers of all solutions: one row per solution, node, chromosome, allele (0 or 1) and ILP
        segment ('Pos', the 'finalPos' of `final_mapping`), with the copy number in 'CN'.
    total_edges : pd.DataFrame
        Edge summary of all solutions: one row per edge and solution, with 'tree_id', 'Parent', 'Child', 'WGD',
        'Deletions' and 'Gains'.
    all_leaves : pd.DataFrame
        Leaf copy numbers of all segments after the first merge ('Pos', the 'mergePos' of `final_mapping`), from
        `process_spice_input`: one row per leaf, chromosome, segment and allele, with 'Start', 'End' and a
        'filter_seg' column that is True for the segments left out of the ILP.
    final_mapping : pd.DataFrame
        Position mapping from `process_spice_input`, linking each 'mergePos' to its 'finalPos'. Modified in place:
        'Allele' is mapped from 'A' / 'B' to 0 / 1.
    maxcn : int
        Maximum copy number, used to compare leaf profiles when choosing each segment's neighbour.
    j : int
        Number of worker processes.
    focal_maxcn : int, optional
        Maximum leaf copy number used in the small parsimony cost. Defaults to `maxcn`.
    logger : logging.Logger, optional
        Logger for progress messages. If None, a logger is created and its handlers are removed at the end.
    allele_specific : bool, default True
        If True, each allele's copy number is modelled separately; if False, total copy number is modelled
        throughout the tree (a dummy second allele is added while filling, see `create_pseudo_allelespecific`).

    Returns
    -------
    final : pd.DataFrame
        Copy numbers of every node at every segment after the first merge: one row per solution, node, chromosome
        and position, with 'tree_id', 'Start', 'End', 'cnA', 'cnB', 'DeletionsA', 'DeletionsB', 'GainsA', 'GainsB'
        and 'cn_model' ('ilp_model' for ILP segments, 'focal_model' for filled ones). In total copy number mode, the
        allele columns are replaced by 'cn', 'Deletions' and 'Gains'.
    total_edges : pd.DataFrame
        `total_edges` with two more columns, the total cost of each solution including the filled segments:
        'Total_cost_focal' (every event) and 'Total_cost_focal_interval' (leaving out events that extend an interval
        already counted at an adjacent segment).
    focal_events : pd.DataFrame
        One row per event at a filled segment in the multiple solutions (see `get_focal_events`).
    """
    final_mapping['Allele'] = final_mapping['Allele'].map({'A': 0, 'B': 1})
    focal_maxcn = maxcn if focal_maxcn is None else focal_maxcn
    if not allele_specific:
        total_nodes_use = create_pseudo_allelespecific(total_nodes)
        all_leaves_use = create_pseudo_allelespecific(all_leaves)
        final_mapping_use = create_pseudo_allelespecific(final_mapping)
    else:
        total_nodes_use = total_nodes
        all_leaves_use = all_leaves
        final_mapping_use = final_mapping
    if logger is None:
        logger, logfile = initialize_logger(True, None)
        close = True
    else:
        close = False
    logger.trace('Running segment tree mapping')
    pos_locs = all_leaves_use[['Chr', 'Pos', 'Start', 'End']].drop_duplicates()

    all_leaves_use = all_leaves_use.drop(['Start', 'End'], axis=1)
    output, states, neighbour_states, total_nodes_mergepos = get_unique_state_trees(total_nodes_use, total_edges,
                                                                                    all_leaves_use, final_mapping_use,
                                                                                    maxcn, j, logger)
    filled_profiles = multi_fill_gaps(states, neighbour_states, j, focal_maxcn=focal_maxcn, weighted=False,
                                      allspec=allele_specific)
    res = (output
           .merge(filled_profiles, on='state_tree_id', how='left')
           [['Solution', 'Node', 'Chr', 'Pos', 'cnA', 'cnB', 'DeletionsA', 'DeletionsB', 'GainsA', 'GainsB']]
           .astype(int, errors='ignore'))
    focal_events = get_focal_events(res, total_edges, allele_specific)
    final = (pd.concat([process_nodes_output(total_nodes_mergepos).assign(cn_model='ilp_model'),
                        res.assign(cn_model='focal_model')])
             .merge(pos_locs, on=['Chr', 'Pos'])
             .sort_values(by=['Solution', 'Node', 'Chr', 'Pos'])
             [['Solution', 'Node', 'Chr', 'Pos', 'Start', 'End', 'cnA', 'cnB', 'DeletionsA', 'DeletionsB', 'GainsA',
               'GainsB', 'cn_model']])
    final.reset_index(drop=True, inplace=True)
    if not allele_specific:
        res = (res[['Solution', 'Node', 'Chr', 'Pos', 'cnA', 'DeletionsA', 'GainsA']]
               .rename(columns={'cnA': 'cn', 'DeletionsA': 'Deletions', 'GainsA': 'Gains'}))
        final = (final[['Solution', 'Node', 'Chr', 'Pos', 'Start', 'End', 'cnA', 'DeletionsA', 'GainsA', 'cn_model']]
                 .rename(columns={'cnA': 'cn', 'DeletionsA': 'Deletions', 'GainsA': 'Gains'}))
    final = total_edges[['tree_id', 'Solution']].drop_duplicates().merge(final, on='Solution', how='right')
    total_edges = (pd.concat([total_edges,
                              add_focal_events_to_edgesdf(res, total_edges),
                              add_focal_events_to_edgesdf(res, total_edges, interval_focal=True, final=final)], axis=1)
                   .drop(['split_group'], axis=1))
    total_edges = total_edges[['tree_id'] + [i for i in total_edges.columns if i != 'tree_id']]
    if close:
        logger.removeHandler(logfile)
        logger.handlers.clear()

    return final, total_edges, focal_events


def process_nodes_output(total_nodes: pd.DataFrame) -> pd.DataFrame:
    """Transpose ILP node copy numbers from one row per allele to one column per allele.

    Parameters
    ----------
    total_nodes : pd.DataFrame
        ILP node copy numbers at the merged segments (from `get_unique_state_trees`), with alleles 0 and 1 and the
        copy number in 'CN'. Modified in place: 'Pos' is cast to int and 'Allele' is mapped to 'A' / 'B'.

    Returns
    -------
    pd.DataFrame
        One row per combination of the remaining columns (solution, node, chromosome, position, ...), with the copy
        number of each allele in 'cnA' and 'cnB', and likewise 'Gains' and 'Deletions' per allele if present.
    """
    total_nodes['Pos'] = total_nodes['Pos'].astype(int)
    total_nodes['Allele'] = total_nodes['Allele'].map({0: 'A', 1: 'B'})
    idx_cols = [i for i in total_nodes.columns if i not in ['Allele', 'CN', 'Gains', 'Deletions']]
    total_nodes = total_nodes.rename(columns={'CN': 'cn'}).pivot(index=idx_cols, columns=['Allele'])
    total_nodes.columns = [''.join(col) for col in total_nodes.columns.values]
    total_nodes.reset_index(inplace=True)
    return total_nodes


def get_unique_state_trees(total_nodes: pd.DataFrame, total_edges: pd.DataFrame, all_leaves: pd.DataFrame,
                           final_mapping: pd.DataFrame, maxcn: int, j: int, logger: logging.Logger
                           ) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Build the state tree of every solution and flagged segment, to be solved by small parsimony.

    Each flagged segment is paired with its most similar kept neighbour (`get_removed_pos_closest_neighbour`). Its
    state tree holds the branch states (parent, child, copy numbers, WGD) of the leaves at the flagged segment and of
    the internal nodes at the neighbour (`get_leaf_node_states`, `multi_get_internal_states`); its neighbour states
    hold the branch states of all nodes at the neighbour. Rows with the same state tree and neighbour states share a
    'state_tree_id', so each distinct state tree is solved only once.

    Parameters
    ----------
    total_nodes : pd.DataFrame
        ILP node copy numbers of all solutions, as in `fill_removed_segments` (two alleles, 0 and 1).
    total_edges : pd.DataFrame
        Edge summary of all solutions: one row per edge and solution, with 'Parent', 'Child' and 'WGD'.
    all_leaves : pd.DataFrame
        Leaf copy numbers of all merged segments, with 'filter_seg', as in `fill_removed_segments` but without
        'Start' and 'End'.
    final_mapping : pd.DataFrame
        Position mapping with 'Chr', 'Allele' (0 or 1), 'finalPos' and 'mergePos'.
    maxcn : int
        Maximum copy number, used to compare leaf profiles when choosing each segment's neighbour.
    j : int
        Number of worker processes.
    logger : logging.Logger
        Logger for progress messages.

    Returns
    -------
    output : pd.DataFrame
        One row per solution and flagged segment ('Solution', 'Chr', 'Pos'), with its 'state_tree_id'.
    states : pd.Series
        State tree of each 'state_tree_id': a sorted tuple of (parent, child, copy numbers, WGD) branch states.
    neighbour_states : pd.Series
        Neighbour states of each 'state_tree_id', in the same form.
    total_nodes_mergepos : pd.DataFrame
        `total_nodes` at the merged segments: one row per kept merged segment ('Pos', the 'mergePos' of
        `final_mapping`) instead of per ILP segment.
    """
    removed_leaves = all_leaves[all_leaves.filter_seg].copy()
    kept_leaves = all_leaves[~all_leaves.filter_seg].copy()
    logger.trace('Getting neighbour CN profiles')
    leaf_distances = get_removed_pos_closest_neighbour(removed_leaves, kept_leaves, maxcn)
    leaf_profiles = get_leaf_node_states(removed_leaves, total_edges)
    total_nodes = total_nodes.merge(final_mapping[['Chr', 'Allele', 'finalPos', 'mergePos']].drop_duplicates(),
                                    left_on=['Chr', 'Allele', 'Pos'], right_on=['Chr', 'Allele', 'finalPos'])
    logger.trace('Getting branch CN states')
    internal_profiles = multi_get_internal_states(total_nodes, total_edges, leaf_distances, j)
    neighbour_profiles = internal_profiles.copy()
    leaf_nodes = all_leaves['Node'].unique()
    total_profiles = pd.concat([internal_profiles[~internal_profiles.Child.isin(leaf_nodes)], leaf_profiles])
    logger.trace('Getting topology CN states')
    neighbour_profiles = (neighbour_profiles
                          .groupby(['Solution', 'Chr', 'Pos'])['Branch_state']
                          .apply(lambda x: tuple(np.sort(x.values)))
                          .reset_index()
                          .rename(columns={'Branch_state': 'Neighbour_state'}))
    total_profiles = (total_profiles
                      .groupby(['Solution', 'Chr', 'Pos'])['Branch_state']
                      .apply(lambda x: tuple(np.sort(x.values)))
                      .reset_index())
    total_profiles = total_profiles.merge(neighbour_profiles, on=['Solution', 'Chr', 'Pos'], how='outer')
    assert total_profiles.isnull().any(axis=1).sum() == 0, \
        'Every flagged segment needs both a state tree and neighbour states (check the neighbour pairing)'
    total_profiles['state_tree_id'] = (total_profiles
                                       .groupby(['Branch_state', 'Neighbour_state'], sort=False)
                                       .ngroup() + 1)
    return (total_profiles[['Solution', 'Chr', 'Pos', 'state_tree_id']],
            total_profiles.drop_duplicates(subset=['state_tree_id']).set_index('state_tree_id')['Branch_state'],
            total_profiles.drop_duplicates(subset=['state_tree_id']).set_index('state_tree_id')['Neighbour_state'],
            total_nodes.drop(['Pos', 'finalPos', 'split_group'], axis=1).rename(columns={'mergePos': 'Pos'}))


def get_removed_pos_closest_neighbour(removed_leaves: pd.DataFrame, kept_leaves: pd.DataFrame,
                                      maxcn: int) -> pd.DataFrame:
    """Pair each flagged segment with the kept segment whose leaf profile is most similar.

    The candidates are the segments just before and just after each run of consecutive flagged segments
    (`get_neighbours`), where they are kept. The leaf profiles (copy number of every leaf and allele) are compared
    with `compare_leaves`. Ties go to the segment before the run.

    Parameters
    ----------
    removed_leaves : pd.DataFrame
        Leaf copy numbers of the flagged segments: one row per leaf, chromosome, segment ('Pos') and allele, with
        'CN'.
    kept_leaves : pd.DataFrame
        Leaf copy numbers of the kept segments, in the same form.
    maxcn : int
        Maximum copy number, used to scale copy number differences in `compare_leaves`.

    Returns
    -------
    pd.DataFrame
        One row per flagged segment ('Chr', 'removed_pos'), with the chosen kept segment in 'neighbours' and its
        distance in 'value'.
    """
    pos_neighbours = (removed_leaves[['Chr', 'Pos']]
                      .drop_duplicates()
                      .groupby('Chr')['Pos']
                      .apply(lambda x: get_neighbours(x.values))
                      .reset_index())
    kept_profiles = (kept_leaves
                     .merge(pos_neighbours[['Chr', 'neighbours']], left_on=['Chr', 'Pos'],
                            right_on=['Chr', 'neighbours'], how='inner')
                     .rename(columns={'Chr': 'Chr_k'})
                     .pivot_table(index=['Chr_k', 'neighbours'], columns=['Node', 'Allele'], values='CN'))
    removed_profiles = (removed_leaves
                        .merge(pos_neighbours[['Chr', 'removed_pos']], left_on=['Chr', 'Pos'],
                               right_on=['Chr', 'removed_pos'], how='inner')
                        .rename(columns={'Chr': 'Chr_r'})
                        .pivot_table(index=['Chr_r', 'removed_pos'], columns=['Node', 'Allele'], values='CN'))
    result = pd.melt(pd.DataFrame(cdist(removed_profiles, kept_profiles, metric=compare_leaves, max_cn=maxcn),
                                  index=removed_profiles.index, columns=kept_profiles.index),
                     ignore_index=False).reset_index()
    result = result[result.Chr_r == result.Chr_k].rename(columns={'Chr_k': 'Chr'}).drop(['Chr_r'], axis=1)
    result = (pos_neighbours
              .merge(result, on=['Chr', 'removed_pos', 'neighbours'], how='inner')
              .sort_values(by=['Chr', 'removed_pos', 'value'], ascending=True)
              .drop_duplicates(subset=['Chr', 'removed_pos'], keep='first'))
    return result


def compare_leaves(profile_1: np.ndarray, profile_2: np.ndarray, max_cn: int) -> float:
    """Return the number of copy numbers that differ, plus their total absolute difference over `max_cn` times the
    profile length."""
    a = np.sum(profile_1 != profile_2)
    b = np.sum(abs(profile_1 - profile_2))
    return a + (b/(max_cn*len(profile_1)))


def get_neighbours(positions: np.ndarray) -> pd.DataFrame:
    """Return the positions just before and just after the run of consecutive positions each position is in."""
    groups = np.split(positions, np.nonzero(np.diff(positions) != 1)[0] + 1)
    return (pd.DataFrame(list(zip(groups, [[np.min(pos)-1, np.max(pos)+1] for pos in groups])),
                         columns=['removed_pos', 'neighbours'])
            .explode('removed_pos')
            .explode('neighbours')
            .set_index('removed_pos'))


def get_leaf_node_states(removed_leaves: pd.DataFrame, total_edges: pd.DataFrame) -> pd.DataFrame:
    """Get the branch state of every leaf at every flagged segment, for every solution.

    Parameters
    ----------
    removed_leaves : pd.DataFrame
        Leaf copy numbers of the flagged segments: one row per leaf, chromosome, segment ('Pos') and allele, with
        'CN'.
    total_edges : pd.DataFrame
        Edge summary of all solutions: one row per edge and solution, with 'Parent', 'Child' and 'WGD'.

    Returns
    -------
    pd.DataFrame
        One row per solution, flagged segment ('Chr', 'Pos') and leaf, with the leaf's branch state in
        'Branch_state': (parent, leaf, copy numbers, WGD), where the copy numbers are a tuple of ints, one per allele
        in row order.
    """
    output = (removed_leaves
              .groupby(['Node', 'Chr', 'Pos'])['CN']
              .agg(lambda x: tuple(x.values.astype(int)))
              .reset_index()
              .rename(columns={'CN': 'State'}))
    output = (output
              .merge(total_edges[['Solution', 'Parent', 'Child', 'WGD']], left_on=['Node'], right_on=['Child'],
                     how='left')
              .set_index(['Solution', 'Chr', 'Pos'])[['Parent', 'Child', 'State', 'WGD']])
    output['Branch_state'] = list(map(tuple, output.values))
    return output['Branch_state'].reset_index()


def multi_get_internal_states(total_nodes: pd.DataFrame, total_edges: pd.DataFrame, leaf_distances: pd.DataFrame,
                              j: int) -> pd.DataFrame:
    """Get the branch state of every node at the chosen neighbour of every flagged segment, for every solution.

    The solutions are split into groups of at most 2000 (about one group per worker), and each group is processed by
    `get_internal_node_states` in a `multiprocessing.Pool`.

    Parameters
    ----------
    total_nodes : pd.DataFrame
        ILP node copy numbers of all solutions, with the merged segment of each ILP segment in 'mergePos'. Modified in
        place: a 'split_group' column is added.
    total_edges : pd.DataFrame
        Edge summary of all solutions: one row per edge and solution, with 'Parent', 'Child' and 'WGD'. Modified in
        place: a 'split_group' column is added.
    leaf_distances : pd.DataFrame
        Chosen neighbour of each flagged segment, from `get_removed_pos_closest_neighbour`.
    j : int
        Number of worker processes.

    Returns
    -------
    pd.DataFrame
        One row per solution, flagged segment ('Chr', 'Pos') and non-root node ('Child'), with the node's branch
        state at the neighbour in 'Branch_state': (parent, child, copy numbers, WGD), where the copy numbers are a
        tuple of ints, one per allele.
    """
    split_size = int(np.minimum(np.ceil(total_edges.Solution.nunique() / j), 2000))
    n_splits = int(np.ceil(total_edges.Solution.nunique() / split_size))
    splits = np.array_split(total_edges.Solution.unique(), n_splits)
    splits = {sol: i for i in range(n_splits) for sol in splits[i]}
    total_nodes['split_group'] = total_nodes['Solution'].map(splits)
    total_edges['split_group'] = total_edges['Solution'].map(splits)
    nodes_list = {k: v for k, v in total_nodes.groupby('split_group')}
    edges_list = {k: v for k, v in total_edges.groupby('split_group')}

    with Pool(processes=j, initializer=init_unique_trees,
              initargs=(nodes_list, edges_list, leaf_distances)) as pool:
        jobs = list(nodes_list.keys())
        bar = ProgressBar(total=len(jobs), length=30, verbose=False)
        progress = (lambda job: bar.progress(advance=True,
                                             msg="Solutions: {}-{}".format((job*split_size)+1,
                                                                           (job*split_size) + split_size)))
        output = [out for sols, out in pool.imap_unordered(get_job_internal_nodes, jobs) if progress(sols)]

    return pd.concat(output).reset_index(drop=True)


def init_unique_trees(_nodes_list: dict, _edges_list: dict, _leaf_distances: pd.DataFrame) -> None:
    """Store the shared inputs of `get_job_internal_nodes` as module-level globals in a worker."""
    global NODES_LIST, EDGES_LIST, LEAF_DISTANCES
    NODES_LIST = _nodes_list
    EDGES_LIST = _edges_list
    LEAF_DISTANCES = _leaf_distances


def get_job_internal_nodes(job: int) -> tuple[int, pd.DataFrame]:
    """Return the group number and the branch states of its solutions (`get_internal_node_states`)."""
    internal_profiles = get_internal_node_states(NODES_LIST[job], EDGES_LIST[job], LEAF_DISTANCES)
    return job, internal_profiles


def get_internal_node_states(total_nodes: pd.DataFrame, total_edges: pd.DataFrame,
                             leaf_distances: pd.DataFrame) -> pd.DataFrame:
    """Get the branch state of every node at the chosen neighbour of every flagged segment, for a group of solutions.

    Parameters
    ----------
    total_nodes : pd.DataFrame
        ILP node copy numbers of the group's solutions, with the merged segment of each ILP segment in 'mergePos'.
    total_edges : pd.DataFrame
        Edge summary of the group's solutions: one row per edge and solution, with 'Parent', 'Child' and 'WGD'.
    leaf_distances : pd.DataFrame
        Chosen neighbour of each flagged segment, from `get_removed_pos_closest_neighbour`.

    Returns
    -------
    pd.DataFrame
        One row per solution, flagged segment ('Chr', 'Pos') and non-root node ('Child'), with the node's branch
        state at the neighbour in 'Branch_state': (parent, child, copy numbers, WGD), where the copy numbers are a
        tuple of ints (allele 0, allele 1).
    """
    use_nodes = (total_nodes
                 .merge(leaf_distances, left_on=['Chr', 'mergePos'], right_on=['Chr', 'neighbours'])
                 [['Solution', 'Node', 'Chr', 'removed_pos', 'Allele', 'CN']])
    use_nodes = use_nodes.merge(total_edges, left_on=['Solution', 'Node'], right_on=['Solution', 'Child'])
    use_nodes = use_nodes.pivot_table(index=['Solution', 'Chr', 'removed_pos', 'Parent', 'Child', 'WGD'],
                                      columns='Allele', values='CN').astype(int)
    use_nodes['CN'] = list(map(tuple, use_nodes.values))
    use_nodes.reset_index(level=['Parent', 'Child', 'WGD'], inplace=True)
    use_nodes['Branch_state'] = list(map(tuple, use_nodes[['Parent', 'Child', 'CN', 'WGD']].values))
    use_nodes.reset_index(inplace=True)
    return (use_nodes[['Solution', 'Chr', 'removed_pos', 'Child', 'Branch_state']]
            .rename(columns={'removed_pos': 'Pos'}))


def multi_fill_gaps(states: pd.Series, neighbour_states: pd.Series, j: int, focal_maxcn: int, weighted: bool = False,
                    allspec: bool = True) -> pd.DataFrame:
    """Solve every unique state tree by small parsimony (`fill_internal_nodes`) in a `multiprocessing.Pool`.

    Parameters
    ----------
    states : pd.Series
        State tree of each 'state_tree_id', from `get_unique_state_trees`.
    neighbour_states : pd.Series
        Neighbour states of each 'state_tree_id', from `get_unique_state_trees`.
    j : int
        Number of worker processes.
    focal_maxcn : int
        Maximum leaf copy number used in the small parsimony cost.
    weighted : bool, default False
        Passed on to `fill_gaps`: if True, copy number changes cost their size instead of 1.
    allspec : bool, default True
        If True, each allele's copy number is modelled separately; if False, total copy number is modelled
        throughout the tree.

    Returns
    -------
    pd.DataFrame
        One row per state tree and node, with 'Node', 'cnA', 'cnB', 'DeletionsA', 'DeletionsB', 'GainsA', 'GainsB'
        and 'state_tree_id', in the order the workers finish.
    """
    with Manager() as manager:
        inputs = manager.dict(states.to_dict())
        nb_inputs = manager.dict(neighbour_states.to_dict())
        with Pool(processes=j,
                  initargs=(inputs, nb_inputs, focal_maxcn, weighted, allspec),
                  initializer=init_fill_gaps) as pool:
            bar = ProgressBar(total=len(states), length=30, verbose=False)
            progress = (lambda e: bar.progress(advance=True, msg="Tree {}".format(e)))
            bar.progress(advance=False, msg="Started")
            results = [result_df.assign(state_tree_id=idx)
                       for idx, result_df in pool.imap_unordered(parallel_fill_gaps, states.index) if progress(idx)]
    return pd.concat(results)


def init_fill_gaps(_inputs: dict, _nb_inputs: dict, _focal_maxcn: int, _weighted: bool, _allspec: bool) -> None:
    """Store the shared inputs of `parallel_fill_gaps` as module-level globals in a worker."""
    global INPUTS, NBINPUTS, FOCALMAXCN, WEIGHTED, ALLSPEC
    INPUTS = _inputs
    NBINPUTS = _nb_inputs
    FOCALMAXCN = _focal_maxcn
    WEIGHTED = _weighted
    ALLSPEC = _allspec


def parallel_fill_gaps(idx: int) -> tuple[int, pd.DataFrame]:
    """Return the state tree ID and its filled copy numbers and events (`fill_internal_nodes`)."""
    result = fill_internal_nodes(INPUTS[idx], NBINPUTS[idx], FOCALMAXCN, WEIGHTED, ALLSPEC)
    return idx, result


def fill_internal_nodes(state_tree: tuple, nb_state_tree: tuple, focal_maxcn: int, weighted: bool,
                        allele_specific: bool) -> pd.DataFrame:
    """Infer the internal node copy numbers and events of one state tree by small parsimony (`fill_gaps`).

    The tree is rebuilt from the branch states, rooted at the child of the ILP root (the most recent common
    ancestor). `fill_gaps` adds a fixed diploid root above it, and the ILP root is added back to the output with
    diploid copy numbers and no events.

    Parameters
    ----------
    state_tree : tuple
        State tree from `get_unique_state_trees`: (parent, child, copy numbers, WGD) branch states of the leaves at
        the flagged segment and of the internal nodes at its neighbour.
    nb_state_tree : tuple
        Branch states of all nodes at the neighbour, in the same form.
    focal_maxcn : int
        Maximum leaf copy number used in the small parsimony cost.
    weighted : bool
        Passed on to `fill_gaps`: if True, copy number changes cost their size instead of 1.
    allele_specific : bool
        If True, each allele's copy number is modelled separately; if False, total copy number is modelled
        throughout the tree (the root is then (2, 0) rather than (1, 1)).

    Returns
    -------
    pd.DataFrame
        One row per node, with 'Node', 'cnA', 'cnB', 'DeletionsA', 'DeletionsB', 'GainsA' and 'GainsB' (events on
        the edge into the node). The ILP root comes last, with index 0.
    """
    parents, children, states, wgd = list(zip(*state_tree))
    _, nb_children, nb_states, _ = list(zip(*nb_state_tree))
    root = [i for i in parents if i not in children][0]
    edges = pd.DataFrame(list(zip(parents, children)), columns=['Parent', 'Child'])
    mrca = edges[edges.Parent == root]['Child'].unique()[0]
    edges = edges[edges.Parent != root]
    edges['Parent'] = edges['Parent'].astype(str)
    edges['Child'] = edges['Child'].astype(str)
    edges['n_events'] = 0
    state_mapping = dict(zip([str(i) for i in children], states))
    nb_state_mapping = dict(zip([str(i) for i in nb_children], nb_states))
    iswgd = dict(zip([str(i) for i in children], wgd))
    make_clade = (lambda node, edges, length=0:
                  Clade(name=node,
                        clades=[make_clade(child['Child'], edges, child['n_events'])
                                for _, child in edges[edges['Parent'] == node].iterrows()],
                        branch_length=length)
                  if node in edges['Parent'].to_list() else Clade(name=node, branch_length=length))

    T = Tree().from_clade(make_clade(mrca, edges))
    node_states, _, gains_losses = fill_gaps(T, state_mapping, nb_state_mapping, weighted=weighted, iswgd=iswgd,
                                             max_cn_eval=focal_maxcn, is_all=allele_specific)
    D = pd.DataFrame(list(map(lambda x: dict(zip(('Node', 'cn', 'Deletions', 'Gains', 'WGD'),
                                                 (x[0], node_states[x[0]]) + x[1])), gains_losses.items())))
    D.index = D.index + 1
    D['cnA'], D['cnB'] = zip(*D['cn'])
    D['DeletionsA'], D['DeletionsB'] = zip(*D['Deletions'])
    D['GainsA'], D['GainsB'] = zip(*D['Gains'])
    if allele_specific:
        D.loc[0] = [root, (1, 1), (0, 0), (0, 0), 0, 1, 1, 0, 0, 0, 0]
    else:
        D.loc[0] = [root, (2, 0), (0, 0), (0, 0), 0, 2, 0, 0, 0, 0, 0]
    return D[['Node', 'cnA', 'cnB', 'DeletionsA', 'DeletionsB', 'GainsA', 'GainsB']]


def fill_gaps(tree: Tree, incns: dict, neighbours: dict, iswgd: Optional[dict] = None, add_diploid_root: bool = True,
              weighted: bool = False, max_cn_eval: int = 20, to_phase: bool = False,
              is_all: bool = True) -> tuple[dict, float, dict]:
    """Infer the internal node copy numbers of a tree by small parsimony (Sankoff), given the leaves and a neighbour.

    The leaf copy numbers are fixed. Each internal node can take any copy number pair (allele A, allele B) whose
    total is at most the largest total in `incns` (after clipping); the internal node values in `incns` only set
    this limit. Costs are computed from the leaves up, then the cheapest states are traced back from the root.

    For each edge and allele, a copy number change costs 0 if the same edge has a change in the same direction that
    is at least as large in `neighbours` (the event then continues from the neighbour). Otherwise it costs 1, or, if
    `weighted` is True, the size of the change minus that of a same-direction change at the neighbour. A child can
    only have copy number above 0 if its parent's is above 0. Among equally costly child states, the one closest to
    the parent is chosen, then the one with the smaller total, then the smaller allele A and allele B.

    Events on each edge are counted against the parent's copy number, doubled if the edge has a WGD: gains are the
    increase and deletions the decrease, per allele.

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        Tree with string node names.
    incns : dict
        Copy numbers of each node, {name: (cn allele A, cn allele B)} as `np.int64`. Only the leaf values (and the
        root's, if `add_diploid_root` is False) are used as fixed values.
    neighbours : dict
        Copy numbers of each node at the neighbouring segment, in the same form.
    iswgd : dict, optional
        Whether the edge into each node has a WGD, {name: bool}. If None, no edge has a WGD.
    add_diploid_root : bool, default True
        If True, a fixed root ('fixed-root') is added above the tree's root, with copy number (1, 1), or (2, 0) if
        `is_all` is False. It is left out of the results.
    weighted : bool, default False
        If True, a copy number change costs its size beyond the neighbour's change instead of 1 (see above).
    max_cn_eval : int, default 20
        Copy numbers are clipped to this value for the cost. The leaves keep their original values in the results.
    to_phase : bool, default False
        If True, each leaf's two alleles may be swapped: both orders cost 0, and the result keeps the order closer to
        the parent's inferred copy number.
    is_all : bool, default True
        If True, each allele's copy number is modelled separately; if False, total copy number is modelled
        throughout the tree (in allele A, with allele B 0).

    Returns
    -------
    result : dict
        Copy numbers of every node of `tree`, {name: (cn allele A, cn allele B)}.
    cost : float
        Total cost of the tree.
    events : dict
        Events on the edge into each node, {name: ((deletions A, deletions B), (gains A, gains B), WGD)}.
    """
    assert (all(isinstance(v, np.int64) for v in incns.values()) or
            all(isinstance(v[0], np.int64) and isinstance(v[1], np.int64) for v in incns.values())), \
        'incns values must be np.int64 copy numbers or pairs of them'
    assert ('fixed-root' not in incns) or (not add_diploid_root), \
        "'fixed-root' is the name of the added diploid root and cannot be a node of the tree"
    T, cns = (Tree(Clade(name='fixed-root', clades=[tree.clade])),
              incns | {'fixed-root': (1, 1) if is_all else (2, 0)}) if add_diploid_root else (tree, incns)
    input_cns = cns
    save_leaves = {leaf.name: input_cns[leaf.name] for leaf in T.clade.get_terminals()}
    input_cns = {v: (min(input_cns[v][0], max_cn_eval), min(input_cns[v][1], max_cn_eval)) for v in input_cns}
    if add_diploid_root:
        neighbours = neighbours | {'fixed-root': (1, 1) if is_all else (2, 0)}
    nb_cns = neighbours
    nb_cns = {v: (min(nb_cns[v][0], max_cn_eval), min(nb_cns[v][1], max_cn_eval)) for v in nb_cns}
    max_cn = max(v[0] + v[1] for v in input_cns.values())

    same_dir = (lambda cnch_diff, cnichi_diff: ((cnch_diff >= 0) & (cnichi_diff >= 0)) |
                                               ((cnch_diff < 0) & (cnichi_diff < 0)))
    calc_diff = (lambda cnch_diff, cnichi_diff:
                 np.maximum(0, abs(cnch_diff) - (abs(cnichi_diff) * same_dir(cnch_diff, cnichi_diff))))

    new_cost_weighted = (lambda cn, ch, cni, chi: calc_diff(cn[0] - ch[0], cni[0] - chi[0]) +
                                                  calc_diff(cn[1] - ch[1], cni[1] - chi[1]))
    new_cost_unweighted = (lambda cn, ch, cni, chi: np.minimum(1, calc_diff(cn[0] - ch[0], cni[0] - chi[0])) +
                                                    np.minimum(1, calc_diff(cn[1] - ch[1], cni[1] - chi[1])))

    def new_cost(cn, ch, cni, chi, weighted=True):
        if weighted:
            cost = new_cost_weighted(cn, ch, cni, chi)
        else:
            cost = new_cost_unweighted(cn, ch, cni, chi)
        return cost

    swap = (lambda P: (P[1], P[0]))
    if to_phase:
        cost = {leaf.name: {input_cns[leaf.name]: 0, swap(input_cns[leaf.name]): 0} for leaf in T.clade.get_terminals()}
        back = {leaf.name: {input_cns[leaf.name]: None, swap(input_cns[leaf.name]): None}
                for leaf in T.clade.get_terminals()}
    else:
        cost = {leaf.name: {input_cns[leaf.name]: 0} for leaf in T.clade.get_terminals()}
        back = {leaf.name: {input_cns[leaf.name]: None} for leaf in T.clade.get_terminals()}
    mkpairs = (lambda M: ((i, n - i) for n in range(M + 1) for i in range(n + 1)))
    it_cns = (lambda n: mkpairs(max_cn) if T.root != n else (input_cns[T.root.name], swap(input_cns[T.root.name])))
    check_loss = (lambda cn, cn_child: (cn[0] > 0 or cn_child[0] == 0) and (cn[1] > 0 or cn_child[1] == 0))

    for node in T.find_clades(order='postorder', terminal=False):
        opts = {cn: {child.name: {cn_child: (cost[child.name][cn_child] + new_cost(cn, cn_child, nb_cns[node.name],
                                                                                   nb_cns[child.name],
                                                                                   weighted=weighted))
                                  if check_loss(cn, cn_child) else np.inf
                                  for cn_child in cost[child.name]}
                     for child in node.clades}
                for cn in it_cns(node)}

        cost[node.name] = {cn: sum(min(opts[cn][child].values()) for child in opts[cn]) for cn in opts}
        back[node.name] = {cn: {child: min(sorted(opts[cn][child].keys(),
                                                  key=(lambda x: (abs(cn[0] - x[0]) + abs(cn[1] - x[1]), x[0] + x[1],
                                                                  x[0], x[1]))),
                                           key=(lambda cn_child: opts[cn][child][cn_child]))
                                for child in opts[cn]}
                           for cn in opts}
    assert not np.isnan(cost[T.root.name][input_cns[T.root.name]]), 'Small parsimony cost of the root is NaN'

    backtrack = (lambda subroot, cn:
                 set.union(*(backtrack(child, back[subroot][cn][child]) for child in back[subroot][cn]))
                 .union({(subroot, cn)})
                 if back[subroot][cn] is not None else {(subroot, cn)})
    result = dict(backtrack(T.root.name, input_cns[T.root.name]))
    if to_phase:
        choose_leaf = (lambda inf, inp: min((inp, swap(inp)), key=(lambda x: abs(x[0] - inf[0]) + abs(x[1] - inf[1]))))
    else:
        choose_leaf = (lambda inf, inp: inp)
    result = {v: result[v] if v not in save_leaves else choose_leaf(result[v], save_leaves[v])
              for v in result if (v != 'fixed-root') or (not add_diploid_root)}
    assert set(result.keys()) == {clade.name for clade in tree.find_clades()}, \
        'Traceback did not assign a copy number to every node of the tree'
    iswgd = {v: False for v in result} if iswgd is None else iswgd

    get_base = (lambda parent, child:
                ((result[parent][0] if parent in result else cns[parent][0]) * (1 + iswgd[child]),
                 (result[parent][1] if parent in result else cns[parent][1]) * (1 + iswgd[child])))
    get_even = (lambda child, base: (
        (0 if base[0] <= result[child][0] else base[0] - result[child][0],
         0 if base[1] <= result[child][1] else base[1] - result[child][1]),
        (0 if base[0] >= result[child][0] else result[child][0] - base[0],
         0 if base[1] >= result[child][1] else result[child][1] - base[1]),
        iswgd[child]))
    events = {child: get_even(child, get_base(parent, child))
              for parent, child in ((node.name, child.name)
                                    for node in T.find_clades(order='preorder', terminal=False)
                                    for child in node.clades)}
    return result, cost[T.root.name][input_cns[T.root.name]], events


def cut_focal_events_by_neighbours(final: pd.DataFrame, res: pd.DataFrame, total_edges: pd.DataFrame) -> pd.DataFrame:
    """Drop the events at filled segments that extend an interval from an adjacent segment.

    An event at a filled segment, on an edge and allele, is dropped (set to 0) if the parent and the child both have
    the same copy number as the segment before, or both as the segment after (in `final`, so either may be an ILP or
    a filled segment). The event is then part of an interval event already counted there.

    Parameters
    ----------
    final : pd.DataFrame
        Copy numbers of every node at every segment, sorted by solution, node, chromosome and position, as returned by
        `fill_removed_segments`.
    res : pd.DataFrame
        Copy numbers and events of every node at every filled segment: one row per solution, node, chromosome and
        position, with a 'Deletions' and 'Gains' column per copy number column of `final`.
    total_edges : pd.DataFrame
        Edge summary of all solutions: one row per edge and solution, with 'Parent' and 'Child'.

    Returns
    -------
    pd.DataFrame
        `res` with the dropped events set to 0, and a 'use_events' column per allele that is False where they were
        dropped (True for nodes without a parent edge).
    """
    cn_cols = [i for i in ['cnA', 'cnB', 'cn'] if i in final.columns]
    alleles = [i.replace('cn', '') for i in cn_cols]
    neighbour_cns = final[['Solution', 'Node', 'Chr', 'Pos', 'cn_model'] + cn_cols].copy()
    neighbour_cns[['leftsame_' + c for c in cn_cols]] = (neighbour_cns
                                                         .groupby(['Solution', 'Node', 'Chr'])[cn_cols]
                                                         .transform(lambda x: np.diff(x, prepend=-1) == 0))
    neighbour_cns[['rightsame_' + c for c in cn_cols]] = (neighbour_cns
                                                          .groupby(['Solution', 'Node', 'Chr'])[cn_cols]
                                                          .transform(lambda x: np.diff(x, append=-1) == 0))
    neighbour_cns = (neighbour_cns[neighbour_cns['cn_model'] == 'focal_model']
                     .drop(cn_cols + ['cn_model'], axis=1).copy())
    test = (total_edges[['Solution', 'Parent', 'Child']]
            .merge(neighbour_cns, left_on=['Solution', 'Parent'], right_on=['Solution', 'Node'], how='left')
            .merge(neighbour_cns, left_on=['Solution', 'Child', 'Chr', 'Pos'],
                   right_on=['Solution', 'Node', 'Chr', 'Pos'], how='left', suffixes=['_P', '_C']))
    for a in alleles:
        test[f'use_events{a}'] = (~((test[f'leftsame_cn{a}_P'] & test[f'leftsame_cn{a}_C']) |
                                    (test[f'rightsame_cn{a}_P'] & test[f'rightsame_cn{a}_C'])))
    test = (test[['Solution', 'Node_C', 'Chr', 'Pos'] + [f'use_events{a}' for a in alleles]]
            .rename(columns={'Node_C': 'Node'}))
    res_test = res.merge(test, on=['Solution', 'Node', 'Chr', 'Pos'], how='left')
    res_test[[f'use_events{a}' for a in alleles]] = (res_test[[f'use_events{a}' for a in alleles]]
                                                     .astype(float)
                                                     .fillna(1)
                                                     .astype(bool))
    for a in alleles:
        res_test[f'Deletions{a}'] = res_test[f'Deletions{a}'] * res_test[f'use_events{a}']
        res_test[f'Gains{a}'] = res_test[f'Gains{a}'] * res_test[f'use_events{a}']
    return res_test


def add_focal_events_to_edgesdf(res: pd.DataFrame, total_edges: pd.DataFrame, interval_focal: bool = False,
                                final: Optional[pd.DataFrame] = None) -> pd.Series:
    """Get the total cost of each solution, including the events at the filled segments.

    The events at the filled segments are summed over alleles and segments for each edge, added to the edge's ILP
    'Deletions' and 'Gains', and summed over the edges of each solution.

    Parameters
    ----------
    res : pd.DataFrame
        Copy numbers and events of every node at every filled segment, as in `cut_focal_events_by_neighbours`.
    total_edges : pd.DataFrame
        Edge summary of all solutions: one row per edge and solution, with 'Child', 'Deletions' and 'Gains'. Not
        modified.
    interval_focal : bool, default False
        If True, leave out the events that extend an interval from an adjacent segment
        (`cut_focal_events_by_neighbours`).
    final : pd.DataFrame, optional
        Copy numbers of every node at every segment, as returned by `fill_removed_segments`. Required if
        `interval_focal` is True.

    Returns
    -------
    pd.Series
        Total cost of the edge's solution, one value per row of `total_edges` (in the same order, with a new
        0, 1, 2, ... index), named 'Total_cost_focal', or 'Total_cost_focal_interval' if `interval_focal` is True.
    """
    cn_cols = [i for i in res.columns if 'cn' in i.lower()]
    alleles = [i.replace('cn', '') for i in cn_cols]
    if interval_focal:
        res = cut_focal_events_by_neighbours(final, res, total_edges)
    focal_events = (res
                    .assign(focal_losses=sum([res[f'Deletions{a}'] for a in alleles]),
                            focal_gains=sum([res[f'Gains{a}'] for a in alleles]))
                    .groupby(['Solution', 'Node'])[['focal_losses', 'focal_gains']]
                    .sum()
                    .reset_index()
                    .rename(columns={'Node': 'Child'}))
    focal_events['Child'] = focal_events['Child'].astype(str)
    total_edges = total_edges.merge(focal_events, on=['Solution', 'Child'], how='left')
    total_edges['Deletions'] = total_edges['Deletions'] + total_edges['focal_losses']
    total_edges['Gains'] = total_edges['Gains'] + total_edges['focal_gains']
    total_edges['Events'] = total_edges['Deletions'] + total_edges['Gains']
    focal_label = '_interval' if interval_focal else ''
    total_edges[f'Total_cost_focal{focal_label}'] = (total_edges.groupby(['Solution'])['Events']
                                                     .transform('sum')
                                                     .astype(int))
    return total_edges[f'Total_cost_focal{focal_label}'].copy()


def get_focal_events(res: pd.DataFrame, total_edges: pd.DataFrame, allele_specific: bool) -> pd.DataFrame:
    """Convert the events at the filled segments of the multiple solutions into one row per event.

    Every event is kept, including those that extend an interval from an adjacent segment, with its full amplitude
    (the change from the parent, doubled on WGD edges). Events on WGD edges are labelled 'Gain' / 'Deletion', not
    '_postWGD', because `fill_gaps` counts them against the doubled parent.

    Parameters
    ----------
    res : pd.DataFrame
        Copy numbers and events of every node at every filled segment: one row per solution, node, chromosome and
        position (segment after the first merge), with 'GainsA', 'GainsB', 'DeletionsA' and 'DeletionsB'.
    total_edges : pd.DataFrame
        Edge summary of all solutions, with 'tree_id' and 'Solution'.
    allele_specific : bool
        If False, only allele A (the total copy number) is used.

    Returns
    -------
    pd.DataFrame
        One row per event in the multiple solutions, with 'tree_id', 'Solution', 'Node', 'Chr', 'startPos' / 'endPos'
        (both the filled segment), 'Allele' (0 or 1), 'Event' ('Gain' or 'Deletion'), 'nEvents' (amplitude) and
        'cn_model' ('focal_model').
    """
    alleles = ['A', 'B'] if allele_specific else ['A']
    events = (res[res['Solution'].str.contains('m')]
              .melt(id_vars=['Solution', 'Node', 'Chr', 'Pos'],
                    value_vars=[f'{e}{a}' for e in ['Gains', 'Deletions'] for a in alleles],
                    var_name='Event', value_name='nEvents')
              .query('nEvents > 0'))
    return (total_edges[['tree_id', 'Solution']].drop_duplicates()
            .merge(events, on='Solution')
            .assign(startPos=lambda df: df['Pos'], endPos=lambda df: df['Pos'],
                    Allele=lambda df: df['Event'].str[-1].map({'A': 0, 'B': 1}),
                    Event=lambda df: df['Event'].str[:-1].map({'Gains': 'Gain', 'Deletions': 'Deletion'}),
                    nEvents=lambda df: df['nEvents'].astype(int), cn_model='focal_model')
            [['tree_id', 'Solution', 'Node', 'Chr', 'startPos', 'endPos', 'Allele', 'Event', 'nEvents', 'cn_model']])


# Event ID from the model and the event's node, chromosome, allele (allele-specific mode only), type and coordinates
# (see `split_ilp_events`)
get_event_id = (lambda df, model, allele_specific:
                (model + '_' + df['Node'].astype(str) + '.' + df['Chr'].astype(str) + '.'
                 + ((df['Allele'].map({0: 'A', 1: 'B'}) + '.') if allele_specific else '')
                 + df['Event'] + '.' + df['Start'].astype(str) + '_' + df['End'].astype(str)))


def split_ilp_events(total_events: pd.DataFrame, pos_mapping: pd.DataFrame, allele_specific: bool) -> pd.DataFrame:
    """Split ILP events at the small segments left out of the ILP, and give each event an 'event_id'.

    An ILP event's 'startPos' / 'endPos' can span small segments that the ILP did not model (filled or removed). Each
    ILP event is therefore cut into one piece per run of ILP bins it spans (an event over bins 1-5 with bin 3 left out
    becomes 1-2 and 4-5), so that every row only spans bins inferred by the ILP. The pieces of an event share its
    'event_id'.

    Each 'event_id' is '<model>_<node>.<chr>.<allele>.<event>.<Start>_<End>', with 'ilp' or 'focal', the allele as
    'A' / 'B', and the coordinates of the event before splitting (the start of its first bin and the end of its last),
    e.g. 'ilp_1.1.A.Gain.1000000_5000000' or 'focal_1.1.A.Gain.3000000_3500000'. In total copy number mode the allele
    is left out ('<model>_<node>.<chr>.<event>.<Start>_<End>'). It is unique within a solution, and the same event in
    different solutions has the same ID. The parts can be recovered for any node name with
    `model, rest = event_id.split('_', 1)`, `node, chr, allele, event, span = rest.rsplit('.', 4)` (or
    `node, chr, event, span = rest.rsplit('.', 3)` in total copy number mode) and `start, end = span.split('_')`.

    Parameters
    ----------
    total_events : pd.DataFrame
        ILP events with 'startPos' / 'endPos' as input bins (from `map_to_original_positions`), with 'Node', 'Chr',
        'Allele' (0 or 1), 'Event', 'Start' and 'End'.
    pos_mapping : pd.DataFrame
        Position mapping from `process_spice_input`, with 'Allele' mapped to 0 / 1; 'finalPos' is NaN for the bins
        left out of the ILP.
    allele_specific : bool
        If True, each allele's copy number is modelled separately and the allele is part of the 'event_id'; if
        False, total copy number is modelled throughout the tree and the allele is left out of it.

    Returns
    -------
    pd.DataFrame
        One row per piece, with the columns of `total_events` (the piece's 'startPos' / 'endPos' and 'Start' / 'End')
        plus 'event_id'.
    """
    kept = (pos_mapping
            .dropna(subset=['finalPos'])[['Chr', 'Allele', 'origPos']]
            .sort_values(by=['Chr', 'Allele', 'origPos']))
    kept['run'] = (kept.groupby(['Chr', 'Allele'])['origPos'].diff() != 1).cumsum()
    runs = (kept
            .groupby(['Chr', 'Allele', 'run'], as_index=False)
            .agg(runStart=('origPos', 'min'), runEnd=('origPos', 'max')))
    bins = pos_mapping[['Chr', 'Allele', 'origPos', 'Start', 'End']]
    return (total_events
            .assign(event_id=lambda df: get_event_id(df, 'ilp', allele_specific))
            .drop(columns=['Start', 'End'])
            .merge(runs, on=['Chr', 'Allele'])
            .query('(runEnd >= startPos) and (runStart <= endPos)')
            .assign(startPos=lambda df: np.maximum(df['startPos'], df['runStart']),
                    endPos=lambda df: np.minimum(df['endPos'], df['runEnd']))
            .merge(bins[['Chr', 'Allele', 'origPos', 'Start']].rename(columns={'origPos': 'startPos'}),
                   on=['Chr', 'Allele', 'startPos'])
            .merge(bins[['Chr', 'Allele', 'origPos', 'End']].rename(columns={'origPos': 'endPos'}),
                   on=['Chr', 'Allele', 'endPos'])
            [list(total_events.columns) + ['event_id']])


def add_focal_events_to_eventsdf(total_events: pd.DataFrame, focal_events: pd.DataFrame,
                                 pos_mapping: pd.DataFrame, allele_specific: bool) -> pd.DataFrame:
    """Combine the ILP events with the events at the filled segments, splitting ILP events at the filled segments.

    The ILP events are split at the filled segments by `split_ilp_events`, so that every row only spans bins inferred
    by its own model; the events at the filled segments get an 'event_id' in the same format, with model 'focal'.

    Parameters
    ----------
    total_events : pd.DataFrame
        ILP events with 'startPos' / 'endPos' as input bins (from `map_to_original_positions`), with 'Node', 'Chr',
        'Allele' (0 or 1), 'Event', 'Start' and 'End'.
    focal_events : pd.DataFrame
        Events at the filled segments, from `get_focal_events`.
    pos_mapping : pd.DataFrame
        Position mapping from `process_spice_input`, with 'Allele' mapped to 0 / 1; 'finalPos' is NaN for the
        filled bins.
    allele_specific : bool
        If True, each allele's copy number is modelled separately and the allele is part of the 'event_id'; if
        False, total copy number is modelled throughout the tree and the allele is left out of it.

    Returns
    -------
    pd.DataFrame
        The ILP pieces and the filled events, with the columns of `total_events` plus 'cn_model' ('ilp_model' or
        'focal_model') and 'event_id', sorted by solution, node, chromosome and 'startPos'.
    """
    cols = list(total_events.columns) + ['cn_model', 'event_id']
    ilp_events = split_ilp_events(total_events, pos_mapping, allele_specific).assign(cn_model='ilp_model')
    focal = (map_to_original_positions(focal_events, pos_mapping, 'mergePos')
             .assign(event_id=lambda df: get_event_id(df, 'focal', allele_specific)))
    return (pd.concat([ilp_events[cols], focal[cols]])
            .sort_values(['Solution', 'Node', 'Chr', 'startPos'], kind='stable',
                         key=lambda col: col.str[1:].astype(int) if col.name == 'Solution' else col)
            .reset_index(drop=True))
