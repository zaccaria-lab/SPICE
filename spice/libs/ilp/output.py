"""Turn a solved SPICE ILP into DataFrames of copy number profiles, events and per-edge summaries.

Variable values are read directly from the Gurobi model (``opt._solver_model``), one row per solution, rounded to the
nearest integer, and matched to the Pyomo variables by their Gurobi names. Pyomo names each variable
``<var>(<index>)`` with the index values joined by underscores, e.g. ``c(1_0_2_3)`` is ``c`` at node 1, allele 0,
chromosome 2, segment 3. Index values must not contain underscores, so leaf profiles need integer chromosome keys.

Solutions are numbered 0 in the optimal ILP, and from 1 in solution pool order in the multiple-solutions ILP.

Variables read
--------------
See ``spice.libs.ilp.optimal.variables`` and ``spice.libs.ilp.multi.variables`` for their full meaning.

c                               : copy number per node and position (``CN``)
g, d, gddot, dddot              : gain / loss amplitude per position; pre- and post-WGD values are summed into
                                  ``Gains`` / ``Deletions`` in the node profiles
gbar, dbar, gddotbar, dddotbar  : number of events per position (``Pos``) in the optimal ILP, and amplitude of the
                                  event per interval (``startPos``, ``endPos``) in the multiple-solutions ILP
w                               : 1 if a WGD occurs on the edge into the node; read from the model in the optimal ILP,
                                  and passed in as ``wgd`` for the multiple-solutions ILP, where ``w`` is a fixed
                                  Param and so has no Gurobi variable

Functions
---------
process_optimal_model_output   : all outputs of the optimal ILP
process_multisol_model_output  : all outputs of the multiple-solutions ILP, for each solution in the pool
get_model_vars                 : values of all variables in one solution
process_vars                   : split variable names into their index values
process_output_cn_profiles     : copy number and gain / loss amplitudes per node, position and solution
process_output_events          : events per node, position or interval, and solution
process_edge_summary           : events, WGD and total cost per edge and solution
"""

from typing import Optional, Sequence

import pandas as pd
import numpy as np
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect

from spice.libs.ilp.utils import get_solution_objective


def process_optimal_model_output(opt: GurobiDirect, tree_edges: Sequence[tuple], debug: bool
                                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Process the solution of the optimal ILP into copy number profiles, events and an edge summary.

    Parameters
    ----------
    opt : pyomo.solvers.plugins.solvers.gurobi_direct.GurobiDirect
        Solver after the solve, giving access to the Gurobi model through ``opt._solver_model``.
    tree_edges : list of tuple
        Tree edges as (parent, child) node IDs.
    debug : bool
        If True, also return the value of every variable.

    Returns
    -------
    total_nodes : pandas.DataFrame
        Copy number profiles, one row per node (including the root), allele and position, with columns
        ``Solution, Node, Chr, Pos, Allele, CN, Deletions, Gains``. ``Gains`` and ``Deletions`` are the summed pre- and
        post-WGD amplitudes at that position. ``Solution`` is always 0.
    total_events : pandas.DataFrame
        All copy number events, one row per non-root node, allele, position and event type with at least one event,
        with columns ``Solution, Node, Chr, Pos, Allele, Event, nEvents``. ``Event`` is ``Gain``, ``Deletion``,
        ``Gain_postWGD`` or ``Deletion_postWGD``, and ``nEvents`` is the number of events of that type starting there.
    edge_summary : pandas.DataFrame
        One row per tree edge, with columns ``Solution, Parent, Child, Events, Deletions, Gains, WGD, Total_cost,
        Gurobi_cost``. ``Events`` is the number of events on the edge, ``Total_cost`` the number of events over the
        whole tree, and ``Gurobi_cost`` the objective value, which also includes the cost of each WGD and a subordinate
        term to break ties (reflecting the weighting of nodes according to depth) (see
        ``spice.libs.ilp.optimal.objective``).
    debug_output : pandas.DataFrame or None
        If ``debug`` is True, the value of every Gurobi variable, indexed by variable name, in a single ``Value``
        column; otherwise None.
    """
    n_solutions = [0]

    varnames = pd.Series(opt._solver_model.getAttr("VarName"))
    varvals = np.vstack([get_model_vars(opt, i) for i in n_solutions])

    total_nodes = process_output_cn_profiles(varnames, varvals, optimal=True)
    total_events = process_output_events(varnames, varvals, optimal=True)
    edge_summary = process_edge_summary(tree_edges, total_events.copy(), varnames=varnames, varvals=varvals,
                                        wgd=None, optimal=True)
    edge_summary['Gurobi_cost'] = opt._solver_model.getAttr('ObjVal')
    total_events = total_events[total_events.nEvents > 0]

    if debug:
        debug_output = pd.DataFrame(varvals, columns=varnames, index=['Value']).T
    else:
        debug_output = None

    return total_nodes, total_events, edge_summary, debug_output


def process_multisol_model_output(opt: GurobiDirect, multi_solutions: bool, tree_edges: Sequence[tuple], debug: bool,
                                  wgd: Optional[dict] = None
                                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Process the solutions of the multiple-solutions ILP into copy number profiles, events and an edge summary.

    Parameters
    ----------
    opt : pyomo.solvers.plugins.solvers.gurobi_direct.GurobiDirect
        Solver after the solve, giving access to the Gurobi model through ``opt._solver_model``.
    multi_solutions : bool
        If True, process every solution in the Gurobi solution pool; otherwise only the best solution.
    tree_edges : list of tuple
        Tree edges as (parent, child) node IDs.
    debug : bool
        If True, also return the value of every variable.
    wgd : dict, optional
        WGD on the edge into each non-root node, {node: 0 or 1}. Must be given when ``w`` is a fixed Param, as it is
        in the multiple-solutions ILP; if None, the WGDs are read from the model's ``w`` variables.

    Returns
    -------
    total_nodes : pandas.DataFrame
        Copy number profiles, one row per solution, node (including the root), allele and position, with columns
        ``Solution, Node, Chr, Pos, Allele, CN, Deletions, Gains``. ``Gains`` and ``Deletions`` are the summed pre- and
        post-WGD amplitudes at that position. Solutions are numbered from 1, in solution pool order which corresponds to
        increasing cost (number of events).
    total_events : pandas.DataFrame
        All copy number events, one row per solution, non-root node, allele, interval and event type with at least one
        event, with columns ``Solution, Node, Chr, startPos, endPos, Allele, Event, nEvents``. ``Event`` is ``Gain``,
        ``Deletion``, ``Gain_postWGD`` or ``Deletion_postWGD``, and ``nEvents`` is the amplitude of that event over
        the interval.
    edge_summary : pandas.DataFrame
        One row per solution and tree edge, with columns ``Solution, Parent, Child, Events, Deletions, Gains, WGD,
        Total_cost, Gurobi_cost``. ``Events`` is the summed amplitude of the events on the edge, ``Total_cost`` the
        same summed over the whole tree, and ``Gurobi_cost`` the objective value of the solution, rounded to a whole
        number, which also includes the cost of each WGD (see ``spice.libs.ilp.multi.objective``).
    debug_output : pandas.DataFrame or None
        If ``debug`` is True, the value of every Gurobi variable in each solution, one row per variable and solution,
        with columns ``index`` (the variable name), ``Solution`` and ``Value``; otherwise None.
    """
    n_solutions = range(1, opt._solver_model.SolCount+1) if multi_solutions else [1]

    varnames = pd.Series(opt._solver_model.getAttr("VarName"))
    varvals = np.vstack([get_model_vars(opt, i-1) for i in n_solutions])
    total_nodes = process_output_cn_profiles(varnames, varvals, optimal=False)
    total_events = process_output_events(varnames, varvals, optimal=False)
    edge_summary = process_edge_summary(tree_edges, total_events.copy(), wgd=wgd, varnames=varnames,
                                        varvals=varvals, optimal=False)
    cost_mapping = {i: get_solution_objective(opt, i-1) for i in n_solutions}
    edge_summary['Gurobi_cost'] = edge_summary.Solution.map(cost_mapping)
    total_events = total_events[total_events.nEvents > 0]

    if debug:
        debug_output = pd.DataFrame(varvals, columns=varnames, index=n_solutions).T
        debug_output = pd.melt(debug_output, ignore_index=False, var_name='Solution', value_name='Value').reset_index()
    else:
        debug_output = None

    return total_nodes, total_events, edge_summary, debug_output


def get_model_vars(opt: GurobiDirect, solution_n: int) -> np.ndarray:
    """Return the values of all variables in the specified solution of the Gurobi solution pool.

    Parameters
    ----------
    opt : pyomo.solvers.plugins.solvers.gurobi_direct.GurobiDirect
        Solver after the solve, giving access to the Gurobi model through ``opt._solver_model``.
    solution_n : int
        Solution number in the pool, 0-based (0 is the best solution found).

    Returns
    -------
    numpy.ndarray
        Value of every Gurobi variable in the solution (``Xn``), in the same order as the variable names
        (``VarName``), rounded to the nearest integer.
    """
    opt._solver_model.params.SolutionNumber = solution_n
    return np.rint(np.array(opt._solver_model.getAttr("Xn")))


def process_vars(var_series: pd.Series, colnames: list[str]) -> pd.DataFrame:
    """Split Gurobi variable names into their index values.

    Parameters
    ----------
    var_series : pandas.Series
        Gurobi variable names of the form ``<var>(<index>)``, with the index values joined by underscores, e.g.
        ``c(1_0_2_3)``.
    colnames : list of str
        Column names: one for the variable name, then one per index value, e.g.
        ``['var', 'Node', 'Allele', 'Chr', 'Pos']``.

    Returns
    -------
    pandas.DataFrame
        One row per variable name, with the same index as ``var_series``, and one string column per name in
        ``colnames``.
    """
    var_df = var_series.str.strip(')').str.replace(r'(', '_').str.split('_', expand=True)
    assert var_df.shape[1] == len(colnames), (f'Variable names split into {var_df.shape[1]} values, expected '
                                              f'{len(colnames)} ({colnames}), e.g. '
                                              f'{var_series[var_series.str.count("_").idxmax()]}. Index values must '
                                              'not contain underscores; use integer chromosome keys in the leaf '
                                              'profiles.')
    var_df.columns = colnames
    return var_df


def process_output_cn_profiles(varnames: pd.Series, varvals: np.ndarray, optimal: bool = True) -> pd.DataFrame:
    """Return the copy number profile of every node, with the gain and loss amplitudes at each position.

    Parameters
    ----------
    varnames : pandas.Series
        Gurobi variable names (``VarName``).
    varvals : numpy.ndarray
        Variable values, one row per solution, in the same order as ``varnames``.
    optimal : bool, optional
        If True, the values come from the optimal ILP and its single solution is numbered 0; otherwise solutions are
        numbered from 1, in solution pool order.

    Returns
    -------
    pandas.DataFrame
        Copy number profiles, one row per solution, node (including the root), allele and position, with columns
        ``Solution, Node, Chr, Pos, Allele, CN, Deletions, Gains`` (all integers). ``Gains`` and ``Deletions`` are the
        summed pre- and post-WGD amplitudes at that position.
    """
    n_sols = 1 if optimal else varvals.shape[0]
    sol_idx = [0] if optimal else range(1, n_sols+1)
    is_c = varnames.str.contains('c(', regex=False)
    c_vars = varnames[is_c]
    c_vars = process_vars(c_vars, ['var', 'Node', 'Allele', 'Chr', 'Pos']).drop('var', axis=1)
    total_nodes = pd.melt(pd.DataFrame(varvals[:, is_c], index=sol_idx, columns=pd.MultiIndex.from_frame(c_vars)).T,
                          ignore_index=False, var_name='Solution', value_name='CN').reset_index()
    is_event = (varnames.str.contains('g(', regex=False) |
                varnames.str.contains('d(', regex=False) |
                varnames.str.contains('gddot(', regex=False) |
                varnames.str.contains('dddot(', regex=False))
    event_vars = varnames[is_event].str.replace('ddot', '')
    event_vars = process_vars(event_vars, ['Event', 'Node', 'Allele', 'Chr', 'Pos'])
    event_vars['Event'] = event_vars['Event'].map({'g': 'Gains', 'd': 'Deletions'})
    total_events = pd.melt(pd.DataFrame(varvals[:, is_event], index=sol_idx,
                                        columns=pd.MultiIndex.from_frame(event_vars)).T,
                           ignore_index=False, var_name='Solution', value_name='nEvents').reset_index()
    total_events = (total_events
                    .pivot_table(index=['Solution', 'Node', 'Allele', 'Chr', 'Pos'], columns='Event', values='nEvents',
                                 aggfunc='sum')
                    .reset_index())
    total_nodes = (total_nodes[['Solution', 'Node', 'Chr', 'Pos', 'Allele', 'CN']]
                   .merge(total_events, on=['Solution', 'Node', 'Chr', 'Pos', 'Allele'], how='left')
                   .fillna(0))
    return total_nodes.astype(int)


def process_output_events(varnames: pd.Series, varvals: np.ndarray, optimal: bool = True) -> pd.DataFrame:
    """Return the copy number events of each type on every non-root node, by position or interval.

    Parameters
    ----------
    varnames : pandas.Series
        Gurobi variable names (``VarName``).
    varvals : numpy.ndarray
        Variable values, one row per solution, in the same order as ``varnames``.
    optimal : bool, optional
        If True, the values come from the optimal ILP and its single solution is numbered 0; otherwise solutions are
        numbered from 1, in solution pool order.

    Returns
    -------
    pandas.DataFrame
        One row per solution, non-root node, allele, position (optimal ILP) or interval (multiple-solutions ILP) and
        event type, including those without an event. The columns are ``Solution, Node, Chr, Pos, Allele, Event,
        nEvents`` in the optimal ILP and ``Solution, Node, Chr, startPos, endPos, Allele, Event, nEvents`` in the
        multiple-solutions ILP, sorted by solution, node, chromosome and (start) position. ``Event`` is ``Gain``,
        ``Deletion``, ``Gain_postWGD`` or ``Deletion_postWGD``. ``nEvents`` (int) is the number of events of that
        type starting at the position in the optimal ILP, and the amplitude of that event over the interval in the
        multiple-solutions ILP.
    """
    n_sols = 1 if optimal else varvals.shape[0]
    sol_idx = [0] if optimal else range(1, n_sols+1)
    is_event = (varnames.str.contains('gbar(', regex=False) |
                varnames.str.contains('dbar(', regex=False) |
                varnames.str.contains('gddotbar(', regex=False) |
                varnames.str.contains('dddotbar(', regex=False))
    event_vars = varnames[is_event]
    use_cols = (['Event', 'Node', 'Allele', 'Chr', 'Pos'] if optimal
                else ['Event', 'Node', 'Allele', 'Chr', 'startPos', 'endPos'])
    event_vars = process_vars(event_vars, use_cols)
    event_vars['Event'] = (event_vars['Event']
                           .map({'gbar': 'Gain', 'dbar': 'Deletion',
                                 'gddotbar': 'Gain_postWGD', 'dddotbar': 'Deletion_postWGD'}))
    total_events = pd.melt(pd.DataFrame(varvals[:, is_event], index=sol_idx,
                                        columns=pd.MultiIndex.from_frame(event_vars)).T,
                           ignore_index=False, var_name='Solution', value_name='nEvents').reset_index()
    index_cols = [i for i in use_cols if i != 'Event']
    total_events[index_cols] = total_events[index_cols].astype(int)
    total_events['nEvents'] = total_events['nEvents'].astype(int)
    if optimal:
        return (total_events[['Solution', 'Node', 'Chr', 'Pos', 'Allele', 'Event', 'nEvents']]
                .sort_values(['Solution', 'Node', 'Chr', 'Pos']))
    else:
        return (total_events[['Solution', 'Node', 'Chr', 'startPos', 'endPos', 'Allele', 'Event', 'nEvents']]
                .sort_values(['Solution', 'Node', 'Chr', 'startPos']))


def process_edge_summary(tree_edges: Sequence[tuple], tev: pd.DataFrame, wgd: Optional[dict] = None,
                         varnames: Optional[pd.Series] = None, varvals: Optional[np.ndarray] = None,
                         optimal: bool = True) -> pd.DataFrame:
    """Return the events, WGD and total cost on every tree edge.

    Parameters
    ----------
    tree_edges : list of tuple
        Tree edges as (parent, child) node IDs.
    tev : pandas.DataFrame
        Copy number events from ``process_output_events``, including those without an event. Modified in place.
    wgd : dict, optional
        WGD on the edge into each non-root node, {node: 0 or 1}. Must be given when ``w`` is a fixed Param, as it is
        in the multiple-solutions ILP; if None, the WGDs are read from the model's ``w`` variables.
    varnames : pandas.Series, optional
        Gurobi variable names (``VarName``). Required, and only used, if ``wgd`` is None.
    varvals : numpy.ndarray, optional
        Variable values, one row per solution, in the same order as ``varnames``. Required, and only used, if ``wgd``
        is None.
    optimal : bool, optional
        If True, the values come from the optimal ILP and its single solution is numbered 0; otherwise solutions are
        numbered from 1, in solution pool order. Only used if ``wgd`` is None.

    Returns
    -------
    pandas.DataFrame
        One row per solution and tree edge, with columns ``Solution, Parent, Child, Events, Deletions, Gains, WGD,
        Total_cost`` (all integers), sorted by solution, parent and child. ``Gains`` and ``Deletions`` are the summed
        ``nEvents`` of the pre- and post-WGD gains and losses on the edge, ``Events`` is their sum, and ``Total_cost``
        is the sum of ``Events`` over the whole tree.
    """
    if wgd is not None:
        wgd_df = pd.DataFrame(wgd, index=['WGD']).T.reset_index().rename(columns={'index': 'Child'}).astype(int)
        merge_cols = ['Child']
    else:
        n_sols = 1 if optimal else varvals.shape[0]
        sol_idx = [0] if optimal else range(1, n_sols+1)
        is_w = varnames.str.contains('w(', regex=False)
        w_vars = varnames[is_w]
        w_vars = process_vars(w_vars, ['var', 'Child']).drop('var', axis=1)
        wgd_df = pd.melt(pd.DataFrame(varvals[:, is_w], index=sol_idx, columns=w_vars['Child']).T, ignore_index=False,
                         var_name='Solution', value_name='WGD').reset_index().astype(int)
        merge_cols = ['Solution', 'Child']
    total_edges = pd.DataFrame(tree_edges, columns=['Parent', 'Child'])
    total_edges = total_edges.merge(wgd_df, on='Child', how='left').fillna(0).sort_values(by=['Parent', 'Child'])
    tev['Event'] = tev['Event'].map({'Gain': 'Gains', 'Deletion': 'Deletions',
                                     'Gain_postWGD': 'Gains', 'Deletion_postWGD': 'Deletions'})
    tev = (tev
           .pivot_table(index=['Solution', 'Node'], columns='Event', values='nEvents', aggfunc='sum', fill_value=0)
           .reset_index()
           .rename(columns={'Node': 'Child'})
           .astype(int))
    tev_cols = [i for i in tev.columns if i not in ['Child', 'Solution']]
    missing_event_cols = [i for i in ['Gains', 'Deletions'] if i not in tev_cols]
    for col in missing_event_cols:
        tev[col] = 0
        tev_cols.append(col)
    total_edges = total_edges.merge(tev, on=merge_cols, how='left').fillna(0).astype(int)
    total_edges['Events'] = total_edges[tev_cols].sum(axis=1)
    total_edges['Total_cost'] = total_edges.groupby('Solution')['Events'].transform('sum')
    return (total_edges[['Solution', 'Parent', 'Child', 'Events'] + tev_cols + ['WGD', 'Total_cost']]
            .sort_values(by=['Solution', 'Parent', 'Child']))
