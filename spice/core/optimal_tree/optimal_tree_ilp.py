import pyomo.environ as pe
import numpy as np
import time
import datetime
import logging
import sys

from spice.core.optimal_tree.model_components import *
from spice.core.auxillary.return_output_n import *
from spice.core.auxillary.ilp_functions import *
from spice.io.logging import *

def get_optimal_tree(leaves, 
                    max_cn, 
                    verbose, 
                    allele_specific, 
                    tree_edges, 
                    time_limit, 
                    weight_wgd,
                    nthreads=1, 
                    mem=0.5,
                    debug=False,
                    rs=1,
                    logger=None,
                    output_dir=None):
    
    if (logger is None):
        logger, logfile = initialize_logger(verbose, output_dir)
        
    logger.trace('Checking optimal tree input')
    leaves = check_leaf_input(leaves, allele_specific, max_cn, logger)
    logger.trace('Input valid')

    logger.trace('Defining ILP')
    # INITIATE MODEL
    model = pe.ConcreteModel()
    model.cons = pe.ConstraintList()
    model, n_loci = add_model_vars(model, leaves, tree_edges, max_cn, allele_specific)
    model = add_model_cons(model, leaves, max_cn, allele_specific)
    node_depth_weighting = get_node_depth_weighting(tree_edges)
    model = add_model_obj(model, max_cn, n_loci, weight_wgd, node_depth_weighting)
    logger.trace('ILP defined')
    logger.info('Model consists of {} variables and {} constraints'.format(get_n_modelvars(model), get_n_modelcons(model)))
    
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
    results = opt.solve(model, tee=verbose, warmstart=True, options={'timelimit':time_limit}, timelimit=time_limit, symbolic_solver_labels=True, load_solutions=True)
    solve_time = time.perf_counter() - solve_start_time
    logger.trace('Processing ILP output')
    
    solve_info = {'optimal_tree_solve_time' : solve_time, 'optimal_tree_model_status' : get_solution_status(opt), 'optimal_tree_gap' : get_gap(results), 'optimal_tree_total_cost' : opt._solver_model.getAttr('ObjVal')}
    solve_info = pd.DataFrame(solve_info, index=['value']).T.reset_index().rename(columns={'index':'parameter'})
    total_nodes, total_events, edge_summary, debug = process_optimal_model_output(opt, tree_edges, debug)
    
    return total_nodes, edge_summary, solve_info, model, debug

def get_node_depth_weighting(tree_edges):
    nodes = set([node for edge in tree_edges for node in edge if node != 0])
    weighting = {j : len(get_ancestor_nodes(j, tree_edges, [], False)) for j in nodes}
    depths = np.unique(list(weighting.values()))
    weights_sum = np.sum(depths.max() - depths + 1)
    depth_weights = dict(zip(depths, (np.max(depths) - depths + 1)/weights_sum))
    return {k: depth_weights[v] for k, v in weighting.items()}
    
def get_ancestor_nodes(node, edges, nodes=[], include_node=True):
    if include_node:
        nodes.append(node)
    parents = [i[0] for i in edges if i[1] == node]
    if len(parents) == 0:
        return nodes
    else:
        nodes = get_ancestor_nodes(parents[0], edges, nodes)
        return nodes
    
    
    
    