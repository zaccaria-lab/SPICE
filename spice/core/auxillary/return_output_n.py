import pandas as pd
import numpy as np
import os
import re
import pyomo.environ as pe

from spice.core.auxillary.ilp_functions import get_solution_objective

def process_optimal_model_output(opt, tree_edges, debug):
    n_solutions = [0]
    
    varnames = pd.Series(opt._solver_model.getAttr("VarName"))
    varvals = np.vstack([get_model_vars(opt, i) for i in n_solutions])
    
    total_nodes = process_output_cn_profiles(varnames, varvals, optimal=True)
    total_events = process_output_events(varnames, varvals, optimal=True)
    edge_summary = process_edge_summary(tree_edges, total_events.copy(), varnames=varnames, varvals=varvals, 
                                        wgd=None, optimal=True)
    edge_summary['Gurobi_cost'] = opt._solver_model.getAttr('ObjVal')
    total_events = total_events[total_events.nEvents > 0]
    
    if debug == True:
        debug_output = pd.DataFrame(varvals, columns=varnames, index=['Value']).T
    else:
        debug_output = None
        
    return total_nodes, total_events, edge_summary, debug_output

def process_multisol_model_output(opt, multi_solutions, tree_edges, debug, wgd=None):
    n_solutions = range(1, opt._solver_model.SolCount+1) if multi_solutions else [1]
    
    varnames = pd.Series(opt._solver_model.getAttr("VarName"))
    varvals = np.vstack([get_model_vars(opt, i-1) for i in n_solutions])
    total_nodes = process_output_cn_profiles(varnames, varvals, optimal=False)
    total_events = process_output_events(varnames, varvals, optimal=False)
    edge_summary = process_edge_summary(tree_edges, total_events.copy(), wgd=wgd, varnames=varnames, 
                                        varvals=varvals, optimal=False)
    cost_mapping = {i : get_solution_objective(opt, i-1) for i in n_solutions}
    edge_summary['Gurobi_cost'] = edge_summary.Solution.map(cost_mapping)
    total_events = total_events[total_events.nEvents > 0]
    
    if debug == True:
        debug_output = pd.DataFrame(varvals, columns=varnames, index=n_solutions).T
        debug_output = pd.melt(debug_output, ignore_index=False, var_name='Solution', value_name='Value').reset_index()
    else:
        debug_output = None
    
    return total_nodes, total_events, edge_summary, debug_output

def get_model_vars(opt, solution_n):
    opt._solver_model.params.SolutionNumber = solution_n
    return np.rint(np.array(opt._solver_model.getAttr("Xn")))

def process_vars(var_series, colnames):
    var_df = var_series.str.strip(')').str.replace(r'(', '_').str.split('_', expand=True)
    var_df.columns = colnames
    return var_df

def process_output_cn_profiles(varnames, varvals, optimal=True):
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
    event_vars['Event'] = event_vars['Event'].map({'g' : 'Gains', 'd' : 'Deletions'})
    total_events = pd.melt(pd.DataFrame(varvals[:, is_event], index=sol_idx, columns=pd.MultiIndex.from_frame(event_vars)).T, 
                           ignore_index=False, var_name='Solution', value_name='nEvents').reset_index()
    total_events = (total_events
                    .pivot_table(index=['Solution', 'Node', 'Allele', 'Chr', 'Pos'], columns='Event', values='nEvents', 
                                 aggfunc='sum')
                    .reset_index())
    total_nodes = (total_nodes[['Solution', 'Node', 'Chr', 'Pos', 'Allele', 'CN']]
                   .merge(total_events, on=['Solution', 'Node', 'Chr', 'Pos', 'Allele'], how='left')
                   .fillna(0))
    return total_nodes.astype(int)

def process_output_events(varnames, varvals, optimal=True):
    n_sols = 1 if optimal else varvals.shape[0]
    sol_idx = [0] if optimal else range(1, n_sols+1)
    is_event = (varnames.str.contains('gbar(', regex=False) | 
                varnames.str.contains('dbar(', regex=False) | 
                varnames.str.contains('gddotbar(', regex=False) | 
                varnames.str.contains('dddotbar(', regex=False))
    event_vars = varnames[is_event]
    use_cols = ['Event', 'Node', 'Allele', 'Chr', 'Pos'] if optimal else ['Event', 'Node', 'Allele', 'Chr', 'startPos', 'endPos']
    event_vars = process_vars(event_vars, use_cols)
    event_vars['Event'] = (event_vars['Event']
                           .map({'gbar': 'Gain', 'dbar': 'Deletion', 'gddotbar': 'Gain_postWGD', 'dddotbar': 'Deletion_postWGD'}))
    total_events = pd.melt(pd.DataFrame(varvals[:, is_event], index=sol_idx, columns=pd.MultiIndex.from_frame(event_vars)).T, 
                           ignore_index=False, var_name='Solution', value_name='nEvents').reset_index()
    total_events[[i for i in use_cols if i != 'Event']] = total_events[[i for i in use_cols if i != 'Event']].astype(int)
    if optimal:
        return (total_events[['Solution', 'Node', 'Chr', 'Pos', 'Allele', 'Event', 'nEvents']]
                .sort_values(['Solution', 'Node', 'Chr', 'Pos']))
    else:
        return (total_events[['Solution', 'Node', 'Chr', 'startPos', 'endPos', 'Allele', 'Event', 'nEvents']]
                .sort_values(['Solution', 'Node', 'Chr', 'startPos']))
    
def process_edge_summary(tree_edges, tev, wgd=None, varnames=None, varvals=None, optimal=True):
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
           .rename(columns={'Node' : 'Child'})
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
