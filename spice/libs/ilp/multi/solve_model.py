"""Solve the SPICE multiple-solutions (enumeration) ILP for one tree topology.

``get_multiple_trees`` builds the model with ``spice.libs.ilp.multi.build_model`` (sets and variables are documented in
``spice.libs.ilp.multi.variables``), warm-started from the optimal ILP's solution, with WGD placement fixed to the
optimal model's values. Gurobi then searches for the ``solution_pool_size`` best solutions, and the solutions are
processed into copy number profiles, events and an edge summary per solution.

Gurobi settings used:

MIPFocus = 1        : focus on finding feasible solutions
PoolSearchMode = 2  : search for the n best solutions (only when more than one solution is requested)
PoolSolutions       : n, the solution pool size
Cutoff              : discard solutions costing more than ``sol_cutoff``. Skipped when a chromosome was reduced to
                      single-segment intervals: events are then counted per independent segment, which inflates the
                      cost well above the optimal cost the cutoff is based on.

Used by ``spice.libs.spice_bootstrap.run_spice_bootstrap``, once per unique topology.
"""

import logging
import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pyomo.environ as pe

from spice.libs.ilp.multi.build_model import add_model_vars, add_model_cons, add_model_obj
from spice.libs.ilp.output import process_multisol_model_output
from spice.libs.ilp.utils import (check_leaf_input, get_model_variables, get_n_modelvars, get_n_modelcons, get_gap,
                                  get_solution_status, get_solution_objective)
from spice.libs.utils.logging import initialize_logger


def get_multiple_trees(leaves: dict,
                       max_cn: int,
                       verbose: bool,
                       allele_specific: bool,
                       tree_edges: Sequence[tuple],
                       time_limit: int,
                       weight_wgd: bool,
                       solution_pool_size: int = 1,
                       init_model: Optional[pe.ConcreteModel] = None,
                       sol_cutoff: Optional[float] = None,
                       nthreads: int = 1,
                       mem: float = 0.5,
                       debug: bool = False,
                       rs: int = 1,
                       logger: Optional[logging.Logger] = None,
                       output_dir: Optional[str] = None,
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Build and solve the multiple-solutions ILP for one tree topology, returning the best solutions found.

    Parameters
    ----------
    leaves : dict
        Leaf copy number profiles in the format {leaf: {chr: {allele: [cn per segment]}}}.
    max_cn : int
        Maximum copy number; also the cost of a WGD when ``weight_wgd`` is True.
    verbose : bool
        If True, print the Gurobi log, Pyomo's solve timings and log at TRACE level.
    allele_specific : bool
        If True, model the copy number of each allele separately; if False, model total copy number throughout the
        tree (the root is then fixed to 2 rather than 1 per allele).
    tree_edges : list of tuple
        Tree edges as (parent, child) node IDs. The root must be node 0.
    time_limit : int
        Gurobi time limit in seconds.
    weight_wgd : bool
        If True, each WGD costs ``max_cn``; if False, it costs 1.
    solution_pool_size : int, optional
        Number of solutions to search for. If 1, only a single solution is returned.
    init_model : pyomo.environ.ConcreteModel, optional
        Solved optimal ILP to warm-start from. Must provide w, because WGD placement is always fixed.
    sol_cutoff : float, optional
        Discard solutions costing more than this. Ignored when a chromosome was reduced to single-segment intervals.
    nthreads : int, optional
        Number of Gurobi threads.
    mem : float, optional
        Gurobi NodefileStart in GB: node files are written to ``output_dir`` beyond this memory use. Only applied when
        ``output_dir`` is given.
    debug : bool, optional
        If True, also return every variable value for every solution.
    rs : int, optional
        Gurobi random seed.
    logger : logging.Logger, optional
        Logger to use; if None, a new one is created with ``initialize_logger``.
    output_dir : str, optional
        Directory for Gurobi node files and, if a new logger is created, the log file.

    Returns
    -------
    total_nodes : pandas.DataFrame
        Copy number of every node, allele, chromosome and position, per solution.
    total_events : pandas.DataFrame
        All copy number events per edge and solution.
    edge_summary : pandas.DataFrame
        Per-edge summary of each solution, including WGD and the Gurobi cost of the solution.
    solve_info : pandas.DataFrame
        Columns 'parameter' and 'value': solve time, model status, gap, total cost range, number of solutions and
        number of optimal solutions.
    debug : pandas.DataFrame or None
        Every variable value per solution if ``debug`` is True, otherwise None.
    """
    if logger is None:
        logger, _ = initialize_logger(verbose, output_dir)

    logger.trace('Checking multiple solutions input')
    leaves = check_leaf_input(leaves, allele_specific, max_cn, logger)
    logger.trace('Input valid')

    logger.trace('Defining ILP')
    # INITIATE MODEL
    model = pe.ConcreteModel()
    model.cons = pe.ConstraintList()

    init_model_vars = get_model_variables(init_model)
    logger.trace('Adding model variables')
    model, _, init_w, extra_segs = add_model_vars(model, leaves, tree_edges, max_cn, init_model=init_model,
                                                  init_model_vars=init_model_vars, logger=logger)
    logger.trace('Adding model constraints')
    model = add_model_cons(model, leaves, max_cn, allele_specific)
    logger.trace('Adding model objective')
    model = add_model_obj(model, max_cn, weight_wgd)
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

    opt.options['MIPFocus'] = 1
    if (sol_cutoff is not None) and (not extra_segs):
        logger.trace(f'Setting solution cutoff to {sol_cutoff} events.')
        opt.options['Cutoff'] = sol_cutoff
    multiple_sols = solution_pool_size > 1
    opt.options['PoolSearchMode'] = int(multiple_sols) * 2
    opt.options['PoolSolutions'] = solution_pool_size

    logger.trace('Running ILP')
    solve_start_time = time.perf_counter()
    results = opt.solve(model, tee=verbose, warmstart=True, options={'timelimit': time_limit}, timelimit=time_limit,
                        symbolic_solver_labels=True, report_timing=verbose)

    solve_time = time.perf_counter() - solve_start_time
    logger.trace('Processing ILP output')
    solve_info = {'multi_solution_solve_time': solve_time, 'multi_solution_model_status': get_solution_status(opt),
                  'multi_solution_gap': get_gap(results),
                  'multi_solution_total_cost_range': (f'{get_solution_objective(opt, 0)}, '
                                                      f'{get_solution_objective(opt, opt._solver_model.SolCount - 1)}'),
                  'n_solutions': opt._solver_model.SolCount,
                  'n_optimal_solutions': len([n for n in range(opt._solver_model.SolCount)
                                              if np.round(get_solution_objective(opt, n), 0)
                                              == np.round(opt._solver_model.getAttr('ObjVal'), 0)])}
    solve_info = pd.DataFrame(solve_info, index=['value']).T.reset_index().rename(columns={'index': 'parameter'})
    total_nodes, total_events, edge_summary, debug = process_multisol_model_output(opt, multiple_sols, tree_edges,
                                                                                  debug, wgd=init_w)
    return total_nodes, total_events, edge_summary, solve_info, debug
