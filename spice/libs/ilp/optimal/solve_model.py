"""Solve the SPICE optimal ILP for one tree topology.

``get_optimal_tree`` builds the model with ``spice.libs.ilp.optimal.build_model`` (sets and variables are documented in
``spice.libs.ilp.optimal.variables``), optionally warm-started from internal node profiles (e.g. from MEDICC), and
solves it with Gurobi for the minimum number of copy number events and WGDs on the tree. The objective's tie-breaking
term weights each non-root node by its depth (``get_node_depth_weighting``): shallower nodes get larger weights, so that
among equally costly solutions, events higher up the tree are preferred.

Gurobi settings used:

Threads        : ``nthreads``
Seed           : ``rs``
TimeLimit      : ``time_limit`` seconds
MIPFocus = 2   : focus on proving optimality.
NodefileStart  : ``mem`` GB; beyond this memory use, node files are written to NodefileDir. Only set when
                 ``output_dir`` is given
NodefileDir    : ``output_dir``

Used by ``spice.libs.spice_bootstrap.run_spice_bootstrap``, once per unique topology. The solved model (after
``check_wgd_presence`` in ``run_spice_bootstrap``) is passed to ``spice.libs.ilp.multi.solve_model.get_multiple_trees``
as ``init_model``, and its cost as the basis of ``sol_cutoff``.
"""

import logging
import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pyomo.environ as pe

from spice.libs.ilp.optimal.build_model import add_model_vars, add_model_cons, add_model_obj
from spice.libs.ilp.output import process_optimal_model_output
from spice.libs.ilp.utils import check_leaf_input, get_n_modelvars, get_n_modelcons, get_gap, get_solution_status
from spice.libs.utils.logging import initialize_logger


def get_optimal_tree(leaves: dict,
                     max_cn: int,
                     verbose: bool,
                     allele_specific: bool,
                     tree_edges: Sequence[tuple],
                     time_limit: int,
                     weight_wgd: bool,
                     init_model: Optional[dict] = None,
                     nthreads: int = 1,
                     mem: float = 0.5,
                     debug: bool = False,
                     rs: int = 1,
                     logger: Optional[logging.Logger] = None,
                     output_dir: Optional[str] = None,
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pe.ConcreteModel, Optional[pd.DataFrame]]:
    """Build and solve the optimal ILP for one tree topology, returning the minimum-cost solution.

    Parameters
    ----------
    leaves : dict
        Leaf copy number profiles in the format {leaf: {chr: {allele: [cn per segment]}}}.
    max_cn : int
        Maximum copy number; also the cost of a WGD when ``weight_wgd`` is True.
    verbose : bool
        If True, print the Gurobi log and log at TRACE level.
    allele_specific : bool
        If True, model the copy number of each allele separately; if False, model total copy number throughout the
        tree (the root is then fixed to 2 rather than 1 per allele).
    tree_edges : list of tuple
        Tree edges as (parent, child) node IDs. The root must be node 0.
    time_limit : int
        Gurobi time limit in seconds.
    weight_wgd : bool
        If True, each WGD costs ``max_cn``; if False, it costs 1.
    init_model : dict, optional
        Start values for c in the format {(node, allele, chr, seg): cn}, e.g. internal node profiles from MEDICC
        (see ``process_medicc_initialise``). If None, c starts at 1 for every internal non-root node (set I).
    nthreads : int, optional
        Number of Gurobi threads.
    mem : float, optional
        Gurobi NodefileStart in GB: node files are written to ``output_dir`` beyond this memory use. Only applied when
        ``output_dir`` is given.
    debug : bool, optional
        If True, also return every variable value of the solution.
    rs : int, optional
        Gurobi random seed.
    logger : logging.Logger, optional
        Logger to use; if None, a new one is created with ``initialize_logger``.
    output_dir : str, optional
        Directory for Gurobi node files and, if a new logger is created, the log file.

    Returns
    -------
    total_nodes : pandas.DataFrame
        Copy number of every node, allele, chromosome and position in the optimal/parsimonious solution, with the
        number of gains and deletions at that position.
    edge_summary : pandas.DataFrame
        Per-edge summary of the solution: number of events, gains and deletions, WGD and total cost, plus the Gurobi
        cost of the solution.
    solve_info : pandas.DataFrame
        Columns 'parameter' and 'value': solve time, model status, gap and total cost (the Gurobi objective value,
        including the fractional tie-breaking term).
    model : pyomo.environ.ConcreteModel
        The solved model, with the solution loaded into its variables.
    debug : pandas.DataFrame or None
        Every variable value of the solution if ``debug`` is True, otherwise None.
    """

    if logger is None:
        logger, _ = initialize_logger(verbose, output_dir)

    logger.trace('Checking optimal tree input')
    leaves = check_leaf_input(leaves, allele_specific, max_cn, logger)
    logger.trace('Input valid')

    logger.trace('Defining ILP')
    # INITIATE MODEL
    model = pe.ConcreteModel()
    model.cons = pe.ConstraintList()
    model, n_loci = add_model_vars(model, leaves, tree_edges, max_cn, init_model)
    model = add_model_cons(model, leaves, max_cn, allele_specific)
    node_depth_weighting = get_node_depth_weighting(tree_edges)
    model = add_model_obj(model, max_cn, n_loci, weight_wgd, node_depth_weighting)
    logger.trace('ILP defined')
    logger.trace('Model consists of {} variables and {} constraints'.format(get_n_modelvars(model),
                                                                            get_n_modelcons(model)))

    # SOLVE MODEL
    opt = pe.SolverFactory('gurobi', solver_io="python")
    opt.options['threads'] = nthreads
    opt.options['seed'] = rs
    if output_dir is not None:
        opt.options['nodefilestart'] = mem
        opt.options['nodefiledir'] = output_dir

    opt.options['MIPFocus'] = 2

    logger.trace('Running ILP')
    solve_start_time = time.perf_counter()
    results = opt.solve(model, tee=verbose, warmstart=True, options={'timelimit': time_limit}, timelimit=time_limit,
                        symbolic_solver_labels=True, load_solutions=True)
    solve_time = time.perf_counter() - solve_start_time
    logger.trace('Processing ILP output')

    solve_info = {'optimal_tree_solve_time': solve_time, 'optimal_tree_model_status': get_solution_status(opt),
                  'optimal_tree_gap': get_gap(results),
                  'optimal_tree_total_cost': opt._solver_model.getAttr('ObjVal')}
    solve_info = pd.DataFrame(solve_info, index=['value']).T.reset_index().rename(columns={'index': 'parameter'})
    total_nodes, _, edge_summary, debug = process_optimal_model_output(opt, tree_edges, debug)

    return total_nodes, edge_summary, solve_info, model, debug


def get_node_depth_weighting(tree_edges: Sequence[tuple]) -> dict:
    """Weight each node by its depth, giving shallower nodes (closer to the root) larger weights.

    A node's depth is its number of ancestors. Each distinct depth d gets the weight (max depth - d + 1), normalised so
    that the weights of the distinct depths sum to 1; every node at depth d gets that weight. Used in the objective's
    tie-breaking term, so that among equally costly solutions, events higher up the tree are preferred.

    Parameters
    ----------
    tree_edges : list of tuple
        Tree edges as (parent, child) node IDs.

    Returns
    -------
    dict
        Weight of each node, {node: weight}. Only nodes in V1 are used by the objective.
    """
    parents, children = list(zip(*tree_edges))
    root = [i for i in parents if i != children]
    nodes = set([node for edge in tree_edges for node in edge if node != root])
    weighting = {j: len(get_ancestor_nodes(j, tree_edges, nodes=[], include_node=False)) for j in nodes}
    depths = np.unique(list(weighting.values()))
    weights_sum = np.sum(depths.max() - depths + 1)
    depth_weights = dict(zip(depths, (np.max(depths) - depths + 1)/weights_sum))
    return {k: depth_weights[v] for k, v in weighting.items()}


def get_ancestor_nodes(node: int, edges: Sequence[tuple], nodes: list = [], include_node: bool = True) -> list:
    """List the ancestors of a node, from its parent up to the root.

    Walks up the tree recursively, appending each ancestor to ``nodes``.

    Parameters
    ----------
    node : int
        Node whose ancestors to list.
    edges : list of tuple
        Tree edges as (parent, child) node IDs.
    nodes : list, optional
        List to append the nodes to (modified in place). The recursive calls pass it on, so every ancestor ends up in
        the same list. If not given, the default list is shared between calls, so callers pass a new list.
    include_node : bool, optional
        If True, append ``node`` itself before its ancestors.

    Returns
    -------
    list
        ``nodes`` with ``node`` (if ``include_node``) and then its ancestors appended, ordered from the node up to
        the root.
    """
    if include_node:
        nodes.append(node)
    parents = [i[0] for i in edges if i[1] == node]
    if len(parents) == 0:
        return nodes
    else:
        nodes = get_ancestor_nodes(parents[0], edges, nodes=nodes)
        return nodes
