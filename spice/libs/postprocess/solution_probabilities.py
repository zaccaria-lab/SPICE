"""Rank the multiple solutions found by SPICE, and give copy number probabilities for each topology.

Each solution (a tree with copy numbers at every node) from the multiple-solutions ILP is given a geometric weight
from its ILP cost ('Total_cost', which leaves out the events of any filled small segments), so that the most
parsimonious solutions dominate. Treating the weights as a distribution over solutions, the module computes for each
solution:

- a topology probability: for each branch, the weighted fraction of solutions in which the child node has the same
  set of ancestors, among the solutions that contain that set of ancestors (summed as logs over the branches);
- a copy number probability: for each internal node and position, the weighted fraction of solutions in which the
  node has the same copy number, among those with the same ancestors and the same ancestral copy numbers (summed as
  logs over the internal nodes and positions);
- a ranking by the sum of the two log probabilities ('solution_rank'), and a ranking of the topologies by their
  topology probability ('topology_rank').

It also gives the bootstrap frequency of each solution's topology, passed in by `run_spice_bootstrap`.

For each topology, it also gives the probability of each copy number at each internal node and position, and marks
the most probable topology and the topology of the top-ranked solution.

`get_solution_probabilities` runs these steps. `run_spice_bootstrap` calls it when `rank_multiple_trees` is True.
"""
import logging
from typing import Optional, Sequence

import pandas as pd
import numpy as np
from scipy.stats import geom

from spice.libs.utils.logging import initialize_logger


def get_solution_probabilities(edges: pd.DataFrame, nodes: pd.DataFrame, bootstrap_frequency: dict,
                               logger: Optional[logging.Logger] = None, cost_col: str = 'Total_cost',
                               allele_specific: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank the multiple solutions, and give copy number probabilities for each topology.

    Each solution is weighted by its cost (`weight_trees`). Its topology probability (`calculate_tree_probabilities`)
    and copy number probability (`calculate_cn_probabilities`) are combined into a ranking, and the topologies are
    ranked by their topology probability (`calculate_tree_rankings`). For each topology, the probability of each copy
    number at each internal node and position is also given (`calculate_cn_position_probabilities`).

    Parameters
    ----------
    edges : pd.DataFrame
        Edges of every solution, with each solution's cost in `cost_col`.
    nodes : pd.DataFrame
        Copy numbers of every node, position and solution.
    bootstrap_frequency : dict
        Fraction of the bootstrap replicates that gave each topology, {tree_id: frequency} (NaN for a fixed
        topology).
    logger : logging.Logger, optional
        SPICE logger. If None, a logger writing to stderr is created, and its handlers are removed at the end.
    cost_col : str
        Column of `edges` with the cost of each solution, used for the weights. SPICE uses the default, 'Total_cost'
        (the number of events in the ILP, leaving out the events of any filled small segments).
    allele_specific : bool
        If True, each allele's copy number is modelled separately; if False, total copy number is modelled
        throughout the tree.

    Returns
    -------
    total_probabilities : pd.DataFrame
        One row per solution, with its geometric weight, its topology and copy number log probabilities, its
        topology's bootstrap frequency, its rank and its topology's rank (1 is the most probable).
    topology_cn_probabilities : pd.DataFrame
        For each topology ('tree_id'), the probability of each copy number at each internal node and position.
        'top_ranked_topology' is True for the most probable topology ('topology_rank' 1; if several topologies share
        rank 1, all of them, with a warning), and 'top_ranked_solution' for the topology of the top-ranked solution
        ('solution_rank' 1; if solutions on several topologies share rank 1, all of them, with a warning); a warning
        is given if the two columns differ.
    """
    if logger is None:
        logger, logfile = initialize_logger(True, None)
        close = True
    else:
        close = False

    logger.trace('Getting unique trees')
    edges = weight_trees(edges, cost_col)
    root = str(edges['Parent'].unique()[~np.isin(edges['Parent'].unique(), edges['Child'].unique())][0])
    c = 'geometric_weight'
    trees = get_unique_trees(edges, tree_weight_col=c)
    logger.trace('Calculating topology probabilities')
    tree_paths = get_edge_paths(trees, tree_weight_col=c)
    tree_probabilities = calculate_tree_probabilities(tree_paths, tree_weight_col=c)

    logger.trace('Calculating copy-number state probabilities')
    nodes = nodes.assign(Node=nodes['Node'].astype(str))
    cn_paths, proc_nodes = get_node_paths(nodes, tree_paths, tree_weight_col=c, allele_specific=allele_specific)
    cn_probabilities = calculate_cn_probabilities(cn_paths, proc_nodes, tree_weight_col=c)

    logger.trace('Calculating tree rankings')
    total_probabilities = calculate_tree_rankings(tree_probabilities, cn_probabilities)
    total_probabilities = (trees
                           .merge(total_probabilities, on=['tree_id', 'Solution'])
                           .merge(edges[['Solution', cost_col, 'Gurobi_cost']].drop_duplicates(), on='Solution')
                           .assign(bootstrap_frequency=lambda x: x['tree_id'].map(bootstrap_frequency))
                           [['tree_id', 'Solution', 'solution_rank', 'topology_rank', 'solution_log_prob',
                             'topology_log_prob', 'cn_log_prob', c, cost_col, 'Gurobi_cost', 'bootstrap_frequency',
                             'Tree']])

    logger.trace('Calculating copy-number state probabilities for each topology')
    top_ranked = total_probabilities.loc[total_probabilities['topology_rank'] == 1, 'tree_id'].unique()
    if len(top_ranked) > 1:
        logger.warning(f'{len(top_ranked)} topologies share the highest topology probability: {sorted(top_ranked)}')
    top_solution_trees = total_probabilities.loc[total_probabilities['solution_rank'] == 1, 'tree_id'].unique()
    if len(top_solution_trees) > 1:
        logger.warning(f'The top-ranked solutions are on {len(top_solution_trees)} topologies: '
                       f'{sorted(top_solution_trees)}')
    if set(top_solution_trees) != set(top_ranked):
        solution_trees = ', '.join(map(str, sorted(top_solution_trees)))
        topology_trees = ', '.join(map(str, sorted(top_ranked)))
        logger.warning(f"The top-ranked solution's topology (tree_id {solution_trees}) is not the most probable "
                       f"topology (tree_id {topology_trees}). This could be due to high uncertainty in the most likely "
                       "topology, review probabilities for details.")
    topology_cn_probabilities = []
    for tree_id in sorted(cn_paths['tree_id'].unique()):
        tree_cn_probabilities = calculate_cn_position_probabilities(cn_paths[cn_paths.tree_id == tree_id], proc_nodes,
                                                                    root, tree_weight_col=c,
                                                                    allele_specific=allele_specific)
        tree_cn_probabilities.insert(0, 'tree_id', tree_id)
        tree_cn_probabilities.insert(1, 'top_ranked_topology', tree_id in top_ranked)
        tree_cn_probabilities.insert(2, 'top_ranked_solution', tree_id in top_solution_trees)
        topology_cn_probabilities.append(tree_cn_probabilities)
    topology_cn_probabilities = pd.concat(topology_cn_probabilities, ignore_index=True)

    if close:
        logger.removeHandler(logfile)
        logger.handlers.clear()

    return total_probabilities, topology_cn_probabilities


def weight_trees(edges: pd.DataFrame, cost_col: str = 'Total_cost') -> pd.DataFrame:
    """Weight each solution by its ILP cost, with geometric weights.

    Each distinct cost gets the geometric probability mass `geom.pmf(k, p=0.8)`, with `k` the cost minus the lowest
    cost plus 1.

    Parameters
    ----------
    edges : pd.DataFrame
        Edges of every solution, with each solution's cost in `cost_col`.
    cost_col : str
        Column of `edges` with the cost of each solution.

    Returns
    -------
    pd.DataFrame
        A copy of `edges` with each solution's weight in 'geometric_weight'.
    """
    edges = edges.copy()
    get_geometric_weight = lambda x: dict(zip(x, geom.pmf(k=[i + 1 - min(x) for i in x], p=0.8)))
    weight_mapping = get_geometric_weight(edges[cost_col].unique())
    edges['geometric_weight'] = edges[cost_col].map(weight_mapping)
    return edges


def get_unique_trees(edges: pd.DataFrame, tree_weight_col: str = 'geometric_weight') -> pd.DataFrame:
    """Give each solution's tree as a sorted tuple of edges, with a topology ID.

    Parameters
    ----------
    edges : pd.DataFrame
        Edges of every solution, with the topology ID in 'tree_id'. Modified in place ('Branch' is added).
    tree_weight_col : str
        Column of `edges` with the weight of each solution.

    Returns
    -------
    pd.DataFrame
        One row per solution, with its tree ('Tree'), weight and topology ID ('tree_id').
    """
    edges['Branch'] = list(zip(edges.Parent.astype(str), edges.Child.astype(str)))
    trees = (edges
             .groupby(['tree_id', 'Solution', tree_weight_col])['Branch']
             .apply(lambda x: tuple(np.sort(x)))
             .reset_index()
             .rename(columns={'Branch': 'Tree'}))
    return trees


def get_edge_paths(trees: pd.DataFrame, tree_weight_col: str = 'geometric_weight') -> pd.DataFrame:
    """Give the ancestors of the child of every edge, for every solution.

    Parameters
    ----------
    trees : pd.DataFrame
        One row per solution, as returned by `get_unique_trees`.
    tree_weight_col : str
        Column of `trees` with the weight of each solution.

    Returns
    -------
    pd.DataFrame
        One row per edge and solution, with the child's ancestors ('NodePath', a tuple sorted by name) and an ID for
        each distinct set of ancestors ('path_id').
    """
    unique_trees = trees[['Tree', 'tree_id']].drop_duplicates()
    res = pd.concat(list(map(get_tree_ancestors, unique_trees.values)))
    unique_paths = {j: i + 1 for i, j in enumerate(res.NodePath.unique())}
    res['path_id'] = list(map(lambda x: unique_paths[x], res['NodePath']))
    trees_res = res.merge(trees[['tree_id', 'Solution', tree_weight_col]], on='tree_id')
    return trees_res


def get_paths(branch: tuple, tree: Sequence[tuple], anc_dict: dict, root: str, mrca: str) -> dict:
    """Add the ancestors of the child of `branch`, and of every edge above it, to `anc_dict`.

    Recurses up the tree until it reaches an edge already in `anc_dict`, or the edge from the root to the MRCA, whose
    child has the root as its only ancestor. The root must have a single child.

    Parameters
    ----------
    branch : tuple
        The edge, as a (parent, child) tuple.
    tree : Sequence[tuple]
        All edges of the tree, as (parent, child) tuples.
    anc_dict : dict
        Ancestors found so far, {edge: the child's ancestors, as a tuple sorted by name}. Modified in place.
    root : str
        Root of the tree.
    mrca : str
        The root's child.

    Returns
    -------
    dict
        `anc_dict`, with `branch` and every edge above it added.
    """
    if branch in anc_dict:
        return anc_dict
    else:
        if branch != (root, mrca):
            parent_branch = [i for i in tree if i[1] == branch[0]][0]
            ancestors = get_paths(parent_branch, tree, anc_dict, root, mrca)
            path = tuple(sorted(set([branch[0]] + list(ancestors[parent_branch]))))
        else:
            path = tuple([branch[0]])
        anc_dict[branch] = path
        return anc_dict


def get_tree_ancestors(tree: Sequence) -> pd.DataFrame:
    """Give the ancestors of the child of every edge of one tree.

    Parameters
    ----------
    tree : Sequence
        The tree's edges and its 'tree_id', as one row of the unique trees in `get_edge_paths`. The root must have a
        single child.

    Returns
    -------
    pd.DataFrame
        One row per edge, with the child's ancestors ('NodePath', a tuple sorted by name).
    """
    tree, tree_id = tree
    ancestors = {}
    parents, children = list(zip(*tree))
    root = [i for i in parents if i not in children][0]
    mrca = [i[1] for i in tree if i[0] == root][0]
    for branch in tree:
        ancestors = get_paths(branch, tree, ancestors, root, mrca)
    out = (pd.Series(ancestors, name='NodePath')
           .reset_index()
           .rename(columns={'level_0': 'Parent', 'level_1': 'Node'})
           .assign(tree_id=tree_id))
    return out.assign(branch=tuple(zip(out.Parent, out.Node)))


def calculate_tree_probabilities(trees_res: pd.DataFrame, tree_weight_col: str = 'geometric_weight') -> pd.DataFrame:
    """Give the log probability of each solution's topology.

    The probability of an edge is the total weight of the solutions in which its child has the same ancestors
    ('path_id'), divided by the total weight of the solutions that have any node with those ancestors. A solution's
    topology probability is the sum of the natural logs of the probabilities of its edges.

    Parameters
    ----------
    trees_res : pd.DataFrame
        One row per edge and solution, as returned by `get_edge_paths`.
    tree_weight_col : str
        Column of `trees_res` with the weight of each solution.

    Returns
    -------
    pd.DataFrame
        One row per solution, with its topology log probability ('topology_log_prob').
    """
    trees_prob = (trees_res[['tree_id', 'Solution', tree_weight_col, 'Node', 'branch', 'path_id']]
                  .drop_duplicates()
                  .copy())
    trees_prob['fract_branch'] = trees_prob.groupby(['path_id', 'Node', 'branch'])[tree_weight_col].transform('sum')
    trees_prob = trees_prob.merge((trees_prob[['path_id', 'Solution', tree_weight_col]]
                                  .drop_duplicates()
                                  .groupby('path_id')[tree_weight_col]
                                  .sum()
                                  .reset_index()
                                  .rename(columns={tree_weight_col: 'fract_path'})), on=['path_id'])
    trees_prob['topology_prob'] = np.where(trees_prob['fract_path'] > 0,
                                           trees_prob['fract_branch'] / trees_prob['fract_path'], 0)
    solution_probabilities = (trees_prob
                              .groupby(['tree_id', 'Solution'])['topology_prob']
                              .apply(lambda x: sum(np.log(list(x))))
                              .rename('topology_log_prob'))
    return solution_probabilities.reset_index()


def get_node_paths(nodes: pd.DataFrame, trees_paths: pd.DataFrame, tree_weight_col: str = 'geometric_weight',
                   allele_specific: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give the copy numbers of the ancestors of every internal node, at every position, for every solution.

    A position is a chromosome and segment, and in allele-specific mode also an allele. An internal node with no copy
    number at a position is given 1.

    Parameters
    ----------
    nodes : pd.DataFrame
        Copy numbers of every node, position and solution ('cnA' and 'cnB' if `allele_specific`, otherwise 'CN').
    trees_paths : pd.DataFrame
        One row per edge and solution, as returned by `get_edge_paths`.
    tree_weight_col : str
        Column of `trees_paths` with the weight of each solution.
    allele_specific : bool
        If True, each allele's copy number is modelled separately; if False, total copy number is modelled
        throughout the tree.

    Returns
    -------
    cn_paths : pd.DataFrame
        One row per solution, internal node (other than the root) and position, with the copy numbers of the node's
        ancestors at that position ('cnPath') and an ID for each distinct 'cnPath' ('cn_path_id').
    nodes : pd.DataFrame
        The rows of `nodes` for the internal nodes, with a 'position' column (one row per allele if
        `allele_specific`).
    """
    internal_nodes = trees_paths.Parent.unique()
    nodes = nodes[nodes.Node.isin(internal_nodes)].copy()
    if allele_specific:
        nodes = pd.melt(nodes, id_vars=['Solution', 'Node', 'Chr', 'Pos', 'Start', 'End'], value_vars=['cnA', 'cnB'],
                        var_name='Allele', value_name='CN')
        nodes['position'] = (nodes['Chr'].astype(str) + '_' + nodes['Pos'].astype(str)
                           + '_' + nodes['Allele'].str.strip('cn'))
    else:
        nodes['position'] = nodes['Chr'].astype(str) + '_' + nodes['Pos'].astype(str)
    nodes_res = nodes.pivot_table(index=['Solution', 'Node'], columns='position', values='CN').fillna(1).astype(int)
    pos = tuple(nodes_res.columns)
    nodes_res = pd.Series(list(map(tuple, nodes_res.values)), index=nodes_res.index, name='CN').reset_index()
    total_res = (nodes_res
                 .rename(columns={'Node': 'NodePath'})
                 .merge(trees_paths[trees_paths.Node.isin(internal_nodes)].explode('NodePath'),
                        on=['Solution', 'NodePath'], how='inner')
                 .sort_values(by=['path_id', 'Solution', 'Node', 'NodePath']))
    cn_paths = (total_res
                .groupby(['Solution', tree_weight_col, 'tree_id', 'path_id', 'Node'])['CN']
                .apply(lambda x: tuple(map(tuple, np.vstack(x).T)))
                .reset_index())
    cn_paths['position'] = [pos]*len(cn_paths)
    cn_paths = cn_paths.explode(['CN', 'position']).reset_index(drop=True).rename(columns={'CN': 'cnPath'})
    cn_path_mapping = dict(zip(np.sort(cn_paths['cnPath'].unique()), range(1, cn_paths['cnPath'].nunique()+1)))
    cn_paths['cn_path_id'] = list(map(lambda x: cn_path_mapping[x], cn_paths['cnPath']))
    cn_paths = cn_paths.merge(trees_paths[['Solution', 'Node', 'NodePath']], on=['Solution', 'Node'], how='left')
    return cn_paths, nodes


def calculate_cn_probabilities(cn_paths: pd.DataFrame, nodes: pd.DataFrame,
                               tree_weight_col: str = 'geometric_weight') -> pd.DataFrame:
    """Give the log probability of each solution's internal node copy numbers.

    The probability of a node's copy number at a position is the total weight of the solutions in which the node has
    that copy number, divided by the total weight of the solutions in which the node has the same ancestors
    ('path_id') with the same copy numbers at that position ('cn_path_id'). A solution's copy number probability is
    the sum of the natural logs of these probabilities over its internal nodes (other than the root) and positions.

    Parameters
    ----------
    cn_paths : pd.DataFrame
        One row per solution, internal node and position, as returned by `get_node_paths`.
    nodes : pd.DataFrame
        Copy numbers of the internal nodes, as returned by `get_node_paths`.
    tree_weight_col : str
        Column of `cn_paths` with the weight of each solution.

    Returns
    -------
    pd.DataFrame
        One row per solution, with its copy number log probability ('cn_log_prob').
    """
    cn_prob = cn_paths.merge((nodes[['Solution', 'Node', 'position', 'CN']]
                              .drop_duplicates()
                              .rename(columns={'CN': 'cn_branch_id'})),
                             on=['Solution', 'Node', 'position'], how='left')
    cn_prob['fract_cn'] = (cn_prob
                           .groupby(['path_id', 'Node', 'position', 'cn_path_id', 'cn_branch_id'])[tree_weight_col]
                           .transform('sum'))
    cn_prob['fract_path'] = (cn_prob
                             .groupby(['path_id', 'Node', 'position', 'cn_path_id'])[tree_weight_col]
                             .transform('sum'))
    cn_prob['cn_prob'] = np.where(cn_prob['fract_path'] > 0, cn_prob['fract_cn'] / cn_prob['fract_path'], 0)
    assert (cn_prob['cn_prob'] <= 1).all(), 'CN probabilities should be between 0 and 1'
    assert (cn_prob['cn_prob'] >= 0).all(), 'CN probabilities should be between 0 and 1'
    assert (cn_prob['fract_path'] >= 0).all(), 'Solution weights should be non-negative'
    cn_probabilities = (cn_prob
                        .groupby(['Solution'])['cn_prob']
                        .apply(lambda x: np.sum(np.log(x)))
                        .rename('cn_log_prob')
                        .reset_index())
    return cn_probabilities


def calculate_cn_position_probabilities(cn_paths: pd.DataFrame, nodes: pd.DataFrame, root: str,
                                        tree_weight_col: str = 'geometric_weight',
                                        allele_specific: bool = True) -> pd.DataFrame:
    """Give the probability of each copy number at each internal node and position, for one topology.

    In each solution, the probability of a node's copy number given its ancestral copy numbers is found as in
    `calculate_cn_probabilities`. The probability of the ancestral copy numbers is the product of the same
    probabilities for the node's ancestors (1 for the root). The product of the two, rounded to 5 decimals, is summed
    over the distinct ancestral copy numbers ('cn_path_id') with which the node has that copy number. The
    probabilities at each node and position must sum to 1 (to 2 decimals).

    Parameters
    ----------
    cn_paths : pd.DataFrame
        One row per solution, internal node and position, as returned by `get_node_paths`, for the solutions of a
        single topology.
    nodes : pd.DataFrame
        Copy numbers of the internal nodes, as returned by `get_node_paths`.
    root : str
        Root of the tree.
    tree_weight_col : str
        Column of `cn_paths` with the weight of each solution.
    allele_specific : bool
        If True, each allele's copy number is modelled separately; if False, total copy number is modelled
        throughout the tree.

    Returns
    -------
    pd.DataFrame
        One row per internal node (other than the root), position and copy number, with its probability
        ('CN_probability').
    """
    allele_col = ['Allele'] if allele_specific else []
    pos_mapping = nodes[['position', 'Chr', 'Pos', 'Start', 'End'] + allele_col].drop_duplicates()
    cn_prob = cn_paths.merge((nodes[['Solution', 'Node', 'position', 'CN']]
                              .drop_duplicates()
                              .rename(columns={'CN': 'cn_branch_id'})),
                             on=['Solution', 'Node', 'position'], how='left')
    cn_prob['fract_cn'] = (cn_prob
                           .groupby(['path_id', 'Node', 'position', 'cn_path_id', 'cn_branch_id'])[tree_weight_col]
                           .transform('sum'))
    cn_prob['fract_path'] = (cn_prob
                             .groupby(['path_id', 'Node', 'position', 'cn_path_id'])[tree_weight_col]
                             .transform('sum'))
    cn_prob['cn_prob'] = np.where(cn_prob['fract_path'] > 0, cn_prob['fract_cn'] / cn_prob['fract_path'], 0)
    assert (cn_prob['cn_prob'] <= 1).all(), 'CN probabilities should be between 0 and 1'
    assert (cn_prob['cn_prob'] >= 0).all(), 'CN probabilities should be between 0 and 1'
    assert (cn_prob['fract_path'] >= 0).all(), 'Solution weights should be non-negative'
    cn_prob_mapping = (cn_prob
                       .groupby(['Solution', 'position'])[['Node', 'cn_prob']]
                       .apply(lambda x: {root: 1} | dict(zip(x.Node, x.cn_prob)))
                       .rename('cn_prob_mapping')
                       .reset_index())
    cn_probabilities = cn_prob.merge(cn_prob_mapping, on=['Solution', 'position'], how='left')
    cn_probabilities['path_prob'] = cn_probabilities.apply(lambda x: np.prod([x['cn_prob_mapping'][node]
                                                                              for node in x['NodePath']]),
                                                           axis=1)
    cn_probabilities['cn_path_prob'] = np.round(cn_probabilities['cn_prob'] * cn_probabilities['path_prob'], 5)
    cn_probabilities = (cn_probabilities[['Node', 'position', 'cn_path_id', 'cn_path_prob', 'cn_branch_id']]
                        .drop_duplicates()
                        .groupby(['Node', 'position', 'cn_branch_id'])['cn_path_prob']
                        .sum()
                        .reset_index()
                        .rename(columns={'cn_branch_id': 'CN', 'cn_path_prob': 'CN_probability'}))
    cn_probabilities = cn_probabilities.merge(pos_mapping, on='position', how='left')
    assert np.all(np.round(cn_probabilities
                           .groupby(['Node', 'Chr', 'Pos'] + allele_col)['CN_probability']
                           .sum()
                           .unique(), 2) == 1), 'Sum of probabilities for each position should be 1.'
    return cn_probabilities[['Node', 'Chr', 'Pos', 'Start', 'End'] + allele_col + ['CN', 'CN_probability']].copy()


def calculate_tree_rankings(tree_probabilities: pd.DataFrame, cn_probabilities: pd.DataFrame) -> pd.DataFrame:
    """Combine the topology and copy number log probabilities of each solution, and rank the solutions and topologies.

    A solution's log probability ('solution_log_prob') is the sum of its topology and copy number log probabilities.
    The solutions are ranked by it ('solution_rank'): 1 is the most probable, solutions with equal values share a
    rank, and no ranks are skipped. The topologies are ranked the same way by their topology log probability
    ('topology_rank'), which is the same for every solution of a topology.

    Parameters
    ----------
    tree_probabilities : pd.DataFrame
        One row per solution, as returned by `calculate_tree_probabilities`.
    cn_probabilities : pd.DataFrame
        One row per solution, as returned by `calculate_cn_probabilities`.

    Returns
    -------
    pd.DataFrame
        One row per solution, with its combined log probability, its rank and its topology's rank, sorted by
        'solution_log_prob' (most probable first).
    """
    total_probabilities = tree_probabilities.merge(cn_probabilities, on='Solution')
    total_probabilities['solution_log_prob'] = (total_probabilities['topology_log_prob']
                                                + total_probabilities['cn_log_prob'])
    total_probabilities = total_probabilities.sort_values(by='solution_log_prob', ascending=False)
    rankings = {j: i+1
                for i, j in enumerate(sorted(total_probabilities['solution_log_prob'].unique(), reverse=True))}
    total_probabilities['solution_rank'] = total_probabilities['solution_log_prob'].map(rankings)
    topology_rankings = {j: i+1
                         for i, j in enumerate(sorted(total_probabilities['topology_log_prob'].unique(),
                                                      reverse=True))}
    total_probabilities['topology_rank'] = total_probabilities['topology_log_prob'].map(topology_rankings)
    return total_probabilities
