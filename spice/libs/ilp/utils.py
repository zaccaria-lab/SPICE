"""Helper functions shared by the SPICE ILPs (``spice.libs.ilp.optimal`` / ``.multi``) and ``spice_bootstrap``.

Input checks
    check_leaf_input            : check the format of the leaf profiles and clip copy numbers above ``max_cn``
    get_seg_pairs               : genomic intervals (p, q) for one chromosome and allele; falls back to single segments
                                  (p, p) when there are more than 1000 pairs

Warm starts
    get_model_variables         : names of the variables in a solved model (multi ILP ``init_model``)
    ensure_initialize_integers  : round warm-start values to integers, keeping only indices in the new model
    _get_init_values            : integer warm-start values for one variable, or None if it is not being initialised
    process_medicc_initialise   : start values for c in the optimal ILP, from MEDICC2 internal node profiles

Solver summaries
    get_n_modelvars             : number of variables in a model
    get_n_modelcons             : number of constraints in a model
    get_gap                     : optimality gap (%) between the upper and lower bound of a solve
    get_solution_objective      : objective value of a solution in the Gurobi solution pool
    get_solution_status         : Gurobi status code as a string (e.g. 'optimal', 'time_limit')

Post-processing (``spice.libs.spice_bootstrap``)
    check_wgd_presence          : remove a WGD called by the model when the child's profile is not doubled across enough
                                  of the genome (inferred)
    check_for_duplicates        : flag solutions with identical edges, node copy numbers and events
    concat_dfs                  : combine the solutions of all topologies, keeping a proportion of each and numbering
                                  them 'm1', 'm2', ... across topologies
    process_output              : keep a proportion of one topology's solutions and renumber them (used by concat_dfs)
    convert_long_to_wide        : transpose the copy number profile Dataframe.
"""

import logging
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import pyomo.environ as pe
from pyomo.opt import SolverResults
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect


def check_leaf_input(leaves: dict, allele_specific: bool, max_cn: int, logger: logging.Logger) -> dict:
    """Check the format of the leaf profiles and clip copy numbers above ``max_cn``.

    Only the first chromosome of the first leaf is checked for the nested dictionary format; every chromosome of every
    leaf is checked for the number of alleles (2 if ``allele_specific``, otherwise 1).

    Parameters
    ----------
    leaves : dict
        Leaf copy number profiles in the format {leaf: {chr: {allele: [cn per segment]}}}.
    allele_specific : bool
        If True, model the copy number of each allele separately; if False, model total copy number throughout the
        tree (the root is then fixed to 2 rather than 1 per allele).
    max_cn : int
        Maximum copy number.
    logger : logging.Logger
        Logger for the warning given when copy numbers are clipped.

    Returns
    -------
    dict
        The leaf profiles, in the same format. If any copy number exceeds ``max_cn``, every profile is clipped to
        [0, ``max_cn``] and returned as numpy arrays; otherwise ``leaves`` is returned unchanged.
    """
    assert (type(leaves) == dict) and (len(leaves) > 0), \
        'Leaves must be a non-empty dictionary in the format {leaf: {chr: {allele: [cn]}}}'
    test_leaf = list(leaves.keys())[0]
    assert (type(leaves[test_leaf]) == dict) and (len(leaves[test_leaf]) > 0), \
        f'Profile of leaf {test_leaf} must be a non-empty dictionary in the format {{chr: {{allele: [cn]}}}}'
    test_chr = list(leaves[test_leaf].keys())[0]
    assert type(leaves[test_leaf][test_chr]) == dict, \
        f'Profile of leaf {test_leaf}, chromosome {test_chr} must be a dictionary in the format {{allele: [cn]}}'
    check_allele_specific = (lambda p, allele_specific: len(p) == allele_specific + 1)
    assert all(check_allele_specific(cns, allele_specific)
               for leaf, profiles in leaves.items() for ch, cns in profiles.items()), \
        ('If running in allele-specific mode, cn format must be {0 : [alleleA cns], 1 : [alleleB cns]} else cn format '
         'must be {0 : [total cns]}')
    input_max_cn = np.max([np.max(allele_cns)
                           for leaf, profiles in leaves.items()
                           for ch, cns in profiles.items()
                           for allele_cns in cns.values()])
    if input_max_cn > max_cn:
        leaves = {leaf: {ch: {allele: np.clip(allele_cns, a_min=0, a_max=max_cn)
                              for allele, allele_cns in cns.items()}
                         for ch, cns in profiles.items()}
                  for leaf, profiles in leaves.items()}
        logger.warning('Input data maximum CN ({}) exceeds specified maximum CN ({}). Clipping CN values to '
                       '{}.'.format(input_max_cn, max_cn, max_cn))
    return leaves


def get_seg_pairs(segs: list[int], ch: int, allele: int, single_seg_chrs: list[tuple],
                  logger: logging.Logger) -> list[tuple]:
    """Return the genomic intervals (p, q), p <= q, on which copy number events can act for one chromosome allele.

    If there are more than 1000 such intervals, only single-segment intervals (p, p) are used instead, and the
    chromosome allele is recorded in ``single_seg_chrs``.

    Parameters
    ----------
    segs : list of int
        Segment numbers (1-based) of the chromosome allele.
    ch : int
        Chromosome.
    allele : int
        Allele (0 or 1).
    single_seg_chrs : list of tuple
        (chr, allele) pairs reduced to single-segment intervals; appended to in place.
    logger : logging.Logger
        Logger used to warn when the chromosome allele is reduced to single-segment intervals.

    Returns
    -------
    list of tuple
        Intervals (p, q) for the chromosome allele.
    """
    seg_pairs = [(p, q) for p in segs for q in segs if p <= q]
    original_n_segs = len(seg_pairs)
    if original_n_segs > 1000:
        seg_pairs = [(p, p) for p in segs]
        single_seg_chrs.append((ch, allele))
        logger.warning(f'Number of segment pairs for chromosome {ch} allele {allele} ({original_n_segs}) exceeds SPICE '
                       f'capacity (1000), using single segments instead - new number of segment pairs: '
                       f'{len(seg_pairs)}')
    return seg_pairs


def ensure_initialize_integers(model_var: dict, keys: Sequence) -> dict:
    """Round warm-start values to integers, keeping only entries whose index is in ``keys``.

    Parameters
    ----------
    model_var : dict
        Variable values from ``var.extract_values()``, mapping index to value (or None).
    keys : sequence
        Indices valid in the new model; any other entries are dropped.

    Returns
    -------
    dict
        Mapping of index to rounded integer value (None values are kept as None).
    """
    return {k: round(v) if v is not None else v for k, v in model_var.items() if k in keys}


def _get_init_values(initialize_model: pe.ConcreteModel, initialize_model_variables: list[str], name: str,
                     keys: Sequence) -> Optional[dict]:
    """Return integer warm-start values for variable ``name``, or None if it is not being initialised.

    Parameters
    ----------
    initialize_model : pyomo.environ.ConcreteModel
        Solved model to take start values from.
    initialize_model_variables : list of str
        Names of variables in ``initialize_model`` to take start values from (see ``get_model_variables``).
    name : str
        Name of the variable, e.g. 'c' or 'w'.
    keys : sequence
        Indices valid in the new model; any other entries are dropped.

    Returns
    -------
    dict or None
        Mapping of index to rounded integer value (see ``ensure_initialize_integers``), or None if ``name`` is not in
        ``initialize_model_variables``.
    """
    if name not in initialize_model_variables:
        return None
    return ensure_initialize_integers(getattr(initialize_model, name).extract_values(), keys)


def get_model_variables(model: Optional[pe.ConcreteModel]) -> list[str]:
    """Return the names of the variables in a model.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel or None
        Solved model to warm-start from, e.g. the optimal ILP passed to the multiple-solutions ILP as ``init_model``.

    Returns
    -------
    list of str
        Names of the Var components with at least one entry, in no fixed order (built from a set); an empty list if
        ``model`` is None.
    """
    if model is None:
        return []
    else:
        block_vars = set([var.parent_component() for var in model.component_data_objects(pe.Var)])
        dc = [str(bv).split('.')[-1] for bv in block_vars]
        return dc


def get_n_modelvars(m: pe.ConcreteModel) -> int:
    """Return the number of variables in a model.

    Parameters
    ----------
    m : pyomo.environ.ConcreteModel
        Model to count the variables of.

    Returns
    -------
    int
        Total number of entries across all Var components, including fixed ones.
    """
    return sum(len(list(var.values())) for var in list(m.component_map(ctype=pe.Var).values()))


def get_n_modelcons(m: pe.ConcreteModel) -> int:
    """Return the number of constraints in a model.

    Parameters
    ----------
    m : pyomo.environ.ConcreteModel
        Model to count the constraints of.

    Returns
    -------
    int
        Total number of entries across all Constraint components, including ConstraintLists such as ``model.cons``.
    """
    return sum(len(list(con.values())) for con in list(m.component_map(ctype=pe.Constraint).values()))


def get_gap(results: SolverResults) -> float:
    """Return the optimality gap of a solve, as a percentage of the upper bound.

    Parameters
    ----------
    results : pyomo.opt.SolverResults
        Results returned by ``opt.solve``.

    Returns
    -------
    float
        (upper bound - lower bound) * 100 / upper bound, rounded to 2 decimal places.
    """
    return np.round((int(results['Problem'][0]['Upper bound']) - int(results['Problem'][0]['Lower bound']))
                    * 100 / int(results['Problem'][0]['Upper bound']), 2)


def get_solution_objective(opt: GurobiDirect, sol: int) -> float:
    """Return the objective value of the specified solution in the Gurobi solution pool.

    Parameters
    ----------
    opt : pyomo.solvers.plugins.solvers.gurobi_direct.GurobiDirect
        Solver after the solve, giving access to the Gurobi model through ``opt._solver_model``.
    sol : int
        Solution number in the pool, 0-based (0 is the best solution found).

    Returns
    -------
    float
        Objective value of the solution (``PoolObjVal``), rounded to 0 decimal places.
    """
    opt._solver_model.params.SolutionNumber = sol
    return np.round(opt._solver_model.getAttr('PoolObjVal'), 0)


def get_solution_status(opt: GurobiDirect) -> Union[str, float]:
    """Return the Gurobi status of the last solve as a string.

    Parameters
    ----------
    opt : pyomo.solvers.plugins.solvers.gurobi_direct.GurobiDirect
        Solver after the solve, giving access to the Gurobi model through ``opt._solver_model``.

    Returns
    -------
    str or float
        'optimal', 'infeasible', 'node_limit', 'time_limit', 'solution_limit' or 'user_interrupt' (Gurobi status
        codes 2, 3, 8, 9, 10 and 11); NaN for any other status.
    """
    status_code = {2: 'optimal', 3: 'infeasible', 8: 'node_limit', 9: 'time_limit', 10: 'solution_limit',
                   11: 'user_interrupt'}
    return (status_code[opt._solver_model.getAttr('Status')]
            if opt._solver_model.getAttr('Status') in [2, 3, 8, 9, 10, 11] else np.nan)


def check_wgd_presence(nodes_data: pd.DataFrame, edges: pd.DataFrame, model: pe.ConcreteModel, seg_sizes: pd.Series,
                       mapping: dict, threshold: float,
                       logger: logging.Logger) -> tuple[pd.DataFrame, pe.ConcreteModel]:
    """Remove WGDs called by the optimal ILP where the child's profile is not doubled across enough of the genome.

    For each edge with a WGD, a position counts as doubled if the child's copy number is at least twice the parent's,
    or zero. The WGD is kept only if, for every allele, the doubled positions cover more than ``threshold`` of the
    allele (weighted by segment size); otherwise it is set to not present in ``edges`` and ``model``.

    Parameters
    ----------
    nodes_data : pd.DataFrame
        Copy number profiles of the optimal solution, one row per node, allele, chromosome and position, with columns
        'Node', 'Allele', 'Chr', 'Pos' and 'CN'. Not modified.
    edges : pd.DataFrame
        Edge summary of the optimal solution, with columns 'Parent', 'Child' and 'WGD'. Modified in place.
    model : pyomo.environ.ConcreteModel
        Solved optimal ILP; ``w`` is set to 0 for each removed WGD.
    seg_sizes : pd.Series
        Segment size ('segsize') indexed by ('Allele', 'Chr', 'Pos').
    mapping : dict
        Mapping of node name to node ID, used to give node names in the log message.
    threshold : float
        Minimum fraction of each allele (or of the genome, in total copy number mode) that must be doubled for a WGD
        to be kept.
    logger : logging.Logger
        Logger for the message given when a WGD is removed.

    Returns
    -------
    edges : pd.DataFrame
        The edge summary with removed WGDs set to 0.
    model : pyomo.environ.ConcreteModel
        The model with ``w`` set to 0 for removed WGDs.
    """
    nodes = nodes_data.copy()
    mapping = {v: k for k, v in mapping.items()}
    parents, children = edges[edges.WGD > 0][['Parent', 'Child']].values.T
    nodes['Allele'] = nodes.Allele.astype(str).str.replace('0', 'A').str.replace('1', 'B')
    nodes['Pos'] = nodes['Pos'].astype(float)
    nodes_table = nodes.pivot_table(index='Node', columns=['Allele', 'Chr', 'Pos'], values='CN')
    diff = pd.DataFrame((nodes_table.reindex(children).values >= (2 * nodes_table.reindex(parents)).values) |
                        (nodes_table.reindex(children).values == 0), index=children, columns=nodes_table.columns).T
    diff_sizes = diff.join(seg_sizes)
    diff_sizes[children] = diff_sizes[children].multiply(diff_sizes['segsize'], axis=0)
    diff_sizes = diff_sizes.reset_index().groupby('Allele').sum()
    prop_true = diff_sizes[children].divide(diff_sizes['segsize'], axis=0)
    wgd_present = (prop_true > threshold).all(axis=0).astype(int)
    edges['WGD'] = (edges['WGD'] * edges['Child'].map(wgd_present).fillna(1)).astype(int)
    for node in wgd_present[wgd_present == 0].index:
        if len(prop_true.index) > 1:
            allele_string = (f'each allele is doubled (A: {round(prop_true[node]["A"]*100)}%, '
                             f'B: {round(prop_true[node]["B"]*100)}%).')
        else:
            allele_string = f'the total copy number profile is doubled ({round(prop_true[node]["A"]*100)}%).'
        logger.info(f'WGD is called by the model for the branch incoming to node {mapping[node]} however less than '
                    f'{threshold*100}% of ' + allele_string + ' Manually correcting WGD to not present.')
        model.w[node] = 0
    return edges, model


def process_medicc_initialise(medicc_nodes: pd.DataFrame, pos_map: pd.DataFrame, node_map: dict,
                              allele_specific: bool) -> dict:
    """Return start values for c in the optimal ILP from MEDICC2 internal node profiles.

    MEDICC2 segments are matched to SPICE positions on chromosome, allele, start and end; segments with no match are
    dropped. Where several MEDICC2 segments map to the same SPICE position, the copy number covering the most base
    pairs is used. Ties go to the copy number closest to diploid (1 per allele, or 2 in total copy number mode), then
    to the smaller one.

    Parameters
    ----------
    medicc_nodes : pd.DataFrame
        MEDICC2 node copy number profiles, with columns 'Node', 'Chr', 'Start', 'End', 'Allele' ('A' or 'B') and 'CN'.
    pos_map : pd.DataFrame
        Mapping of input segments to SPICE positions, with columns 'Chr', 'Allele', 'Start', 'End' and 'finalPos'.
    node_map : dict
        Mapping of node name to node ID.
    allele_specific : bool
        If True, each allele's copy number is modelled separately; if False, total copy number is modelled throughout
        the tree. Sets the diploid copy number used to break ties.

    Returns
    -------
    dict
        Start values for c, {(node, allele, chr, seg): cn}.
    """
    diploid_cn = 1 if allele_specific else 2
    internal_nodes_pos = (medicc_nodes
                          .merge(pos_map[['Chr', 'Allele', 'Start', 'End', 'finalPos']],
                                 on=['Chr', 'Allele', 'Start', 'End'], how='left')
                          .dropna(subset=['finalPos'])
                          .astype({'finalPos': int})
                          .assign(seg_size=lambda df: df['End'] - df['Start']))
    internal_nodes_pos['Allele'] = internal_nodes_pos['Allele'].map({'A': 0, 'B': 1})
    internal_nodes_pos['Node'] = internal_nodes_pos['Node'].map(node_map)
    pos_cols = ['Node', 'Allele', 'Chr', 'finalPos']
    cn_sizes = (internal_nodes_pos
                .groupby(pos_cols + ['CN'], as_index=False)['seg_size'].sum()
                .assign(diploid_dist=lambda df: (df['CN'] - diploid_cn).abs()))
    spice_init_dict = (cn_sizes
                       .sort_values(by=['seg_size', 'diploid_dist', 'CN'], ascending=[False, True, True])
                       .drop_duplicates(subset=pos_cols)
                       .set_index(pos_cols)['CN']
                       .to_dict())
    return spice_init_dict


def check_for_duplicates(total_nodes: pd.DataFrame, total_edges: pd.DataFrame, total_events: pd.DataFrame,
                         logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flag solutions that are identical in their edges, node copy numbers and events.

    Two solutions are duplicates if all their rows in ``total_edges`` and ``total_events`` are equal, in order, and the
    copy number columns ('cnA' / 'cnB', or 'cn' in total copy number mode) of ``total_nodes`` are equal once sorted
    by node, chromosome and position.

    Parameters
    ----------
    total_nodes : pd.DataFrame
        Node copy numbers of all solutions in wide format (see ``convert_long_to_wide``): one row per tree, solution,
        node and position, with one copy number column per allele.
    total_edges : pd.DataFrame
        Edge summary of all solutions: one row per edge and solution.
    total_events : pd.DataFrame
        All copy number events per edge and solution.
    logger : logging.Logger
        Logger for the number of duplicate solutions found.

    Returns
    -------
    total_nodes, total_edges, total_events : pd.DataFrame
        The inputs. If any duplicates are found, a 'dup_idx' column is added to each (in place), giving the same
        'idx_<n>' label to solutions that are duplicates of each other and NaN to the rest; otherwise they are
        returned unchanged.
    """
    cn_cols = [i for i in ['cnA', 'cnB', 'cn'] if i in total_nodes.columns]
    edge_cols = [i for i in total_edges.columns if i != 'Solution']
    event_cols = [i for i in total_events.columns if i != 'Solution']
    comp_te = (total_edges
               .groupby('Solution')[edge_cols]
               .apply(lambda x: tuple(map(tuple, x.values))))
    comp_tn = (total_nodes
               .sort_values(by=['Solution', 'Node', 'Chr', 'Pos'])
               .groupby('Solution')[cn_cols].apply((lambda x: tuple(map(tuple, x.values))))
               .sort_values())
    comp_tev = (total_events
                .groupby('Solution')[event_cols]
                .apply(lambda x: tuple(map(tuple, x.values))))
    dups = (pd.concat([comp_te, comp_tn, comp_tev], axis=1)
            .reset_index()
            .groupby([0, 1, 2])['Solution']
            .agg(Solution=list, n='count')
            .reset_index()[['Solution', 'n']]
            .sort_values(by='n', ascending=False))
    if dups['n'].max() == 1:
        logger.info('No duplicate solutions found')
        return total_nodes, total_edges, total_events
    else:
        logger.warning(f'Found {dups[dups.n > 1]["n"].sum()} duplicate solutions')
        dups['dup_idx'] = (pd.Series([f'idx_{i}' for i in range(1, len(dups) + 1)], index=dups.index)
                           .where(dups['n'] > 1))
        dup_mapping = dups.explode('Solution').set_index('Solution')['dup_idx'].to_dict()
        total_nodes['dup_idx'] = total_nodes['Solution'].map(dup_mapping)
        total_edges['dup_idx'] = total_edges['Solution'].map(dup_mapping)
        total_events['dup_idx'] = total_events['Solution'].map(dup_mapping)
        return total_nodes, total_edges, total_events


def concat_dfs(solution_list: list[pd.DataFrame], prop_sols_used: float) -> pd.DataFrame:
    """Combine the solutions of all topologies, keeping the required proportion of each (that maintains consistent
    frequencies per topology) and numbering them across topologies.

    Parameters
    ----------
    solution_list : list of pd.DataFrame
        One DataFrame per topology, with a 'Solution' column of 1-based solution numbers.
    prop_sols_used : float
        Proportion of each topology's solutions to keep (see ``process_output``).

    Returns
    -------
    pd.DataFrame
        The kept solutions of all topologies, with 'Solution' renamed to 'm1', 'm2', ... continuing across topologies,
        and a fresh index.
    """
    processed_list = []
    solution_count = 0
    for i in solution_list:
        processed_sols, solution_count = process_output(i, prop_sols_used, solution_count)
        processed_list.append(processed_sols)
    return pd.concat(processed_list).reset_index(drop=True)


def process_output(solutions: pd.DataFrame, prop_sols_used: float, solution_count: int = 0) -> tuple[pd.DataFrame, int]:
    """Keep a proportion of one topology's solutions and renumber them after the solutions already kept.

    The first round(n * ``prop_sols_used``) solutions are kept (at least one), where n is the number of solutions.

    Parameters
    ----------
    solutions : pd.DataFrame
        Solutions of one topology, with a 'Solution' column of 1-based solution numbers.
    prop_sols_used : float
        Proportion of the solutions to keep.
    solution_count : int, optional
        Number of solutions already kept from earlier topologies (default: 0).

    Returns
    -------
    solutions : pd.DataFrame
        The kept solutions, with 'Solution' renamed to 'm<solution_count + n>'.
    solution_count : int
        ``solution_count`` plus the number of solutions kept.
    """
    max_sol_n = np.maximum(round(solutions['Solution'].nunique() * prop_sols_used), 1)
    solutions = solutions[solutions.Solution <= max_sol_n].copy()
    solutions['Solution'] = 'm' + (solutions['Solution'] + solution_count).astype(str)
    solution_count += max_sol_n
    return solutions, solution_count


def convert_long_to_wide(total_nodes: pd.DataFrame) -> pd.DataFrame:
    """Transpose node copy numbers from one row per allele to one column per allele.

    Parameters
    ----------
    total_nodes : pd.DataFrame
        Node copy numbers in long format: one row per tree, solution, node, position and allele, with columns
        'tree_id', 'Solution', 'Node', 'Chr', 'Pos', 'Start', 'End', 'Allele' (0 or 1), 'CN', 'Gains' and
        'Deletions'. Its 'Allele' column is changed to 'A' / 'B' in place.

    Returns
    -------
    pd.DataFrame
        Node copy numbers in wide format: one row per tree, solution, node and position, with columns 'cnA', 'cnB',
        'DeletionsA', 'DeletionsB', 'GainsA' and 'GainsB' if there are two alleles, or 'cn', 'Deletions' and 'Gains'
        if there is one.
    """
    total_nodes['Allele'] = total_nodes['Allele'].map({0: 'A', 1: 'B'})
    alleles = total_nodes.Allele.unique()
    cn = (total_nodes
          .rename(columns={'CN': 'cn'})
          .pivot(index=['tree_id', 'Solution', 'Node', 'Chr', 'Pos', 'Start', 'End'],
                 columns='Allele', values=['cn', 'Deletions', 'Gains']))
    if len(alleles) == 1:
        cn.columns = [col[0] for col in cn.columns.values]
    else:
        cn.columns = [''.join(col) for col in cn.columns.values]
    return cn.reset_index()
