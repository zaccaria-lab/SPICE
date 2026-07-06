import pyomo.environ as pe
import numpy as np
import time
import datetime
import logging
import sys

from spice.core.multi_solutions.model_components import *
from spice.core.auxillary.return_output_n import *
from spice.core.auxillary.ilp_functions import *
from spice.io.logging import *

def get_multiple_trees(leaves, 
                        max_cn, 
                        verbose, 
                        allele_specific, 
                        tree_edges, 
                        time_limit, 
                        weight_wgd,
                        fix_wgd=False,
                        solution_pool_size=1,
                        init_model=None,
                        sol_cutoff=None,
                        nthreads=1, 
                        mem=0.5,
                        debug=False,
                        rs=1,
                        logger=None,
                        output_dir=None):
    
    if (logger is None):
        logger, logfile = initialize_logger(verbose, output_dir)
        
    logger.trace('Checking multiple solutions input')
    leaves = check_leaf_input(leaves, allele_specific, max_cn, logger)
    logger.trace('Input valid')

    logger.trace('Defining ILP')
    start = time.perf_counter()
    # INITIATE MODEL
    model = pe.ConcreteModel()
    model.cons = pe.ConstraintList()

    init_model_vars = get_model_variables(init_model)
    logger.trace('Adding model variables')
    model, n_loci, init_w, extra_segs = add_model_vars(model, leaves, tree_edges, max_cn, allele_specific, init_model, 
                                                       init_model_vars, fix_wgd, logger)
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
    if (sol_cutoff is not None) and (extra_segs == False):
        logger.trace(f'Setting solution cutoff to {sol_cutoff} events.')
        opt.options['Cutoff'] = sol_cutoff
    multiple_sols  = True if solution_pool_size > 1 else False
    opt.options['PoolSearchMode'] = int(multiple_sols) * 2
    opt.options['PoolSolutions'] = solution_pool_size   
    
    logger.trace('Running ILP')
    solve_start_time = time.perf_counter()
    results = opt.solve(model, tee=verbose, warmstart=True, options={'timelimit':time_limit}, timelimit=time_limit, 
                        symbolic_solver_labels=True, report_timing=True)

    solve_time = time.perf_counter() - solve_start_time
    logger.trace('Processing ILP output')
    solve_info = {'multi_solution_solve_time' : solve_time, 'multi_solution_model_status' : get_solution_status(opt), 
                  'multi_solution_gap' : get_gap(results), 
                  'multi_solution_total_cost_range' : f'{get_solution_objective(opt, 0)}, \
                                                        {get_solution_objective(opt, opt._solver_model.SolCount-1)}',
                  'n_solutions' : opt._solver_model.SolCount, 
                  'n_optimal_solutions' : len([n for n in range(opt._solver_model.SolCount) 
                                               if np.round(get_solution_objective(opt, n), 0) == np.round(opt._solver_model.getAttr('ObjVal'), 0)])}
    solve_info = pd.DataFrame(solve_info, index=['value']).T.reset_index().rename(columns={'index':'parameter'})
    wgd_vals = init_w if fix_wgd else None
    print('starting process output', datetime.datetime.now())
    total_nodes, total_edges, edge_summary, debug = process_multisol_model_output(opt, multiple_sols, tree_edges, debug, 
                                                                                  wgd=init_w)
    print('process output complete', datetime.datetime.now())
    return total_nodes, total_edges, edge_summary, solve_info, debug

    