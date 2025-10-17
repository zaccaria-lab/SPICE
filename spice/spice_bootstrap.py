import pyomo.environ as pe
import numpy as np
import datetime
import os, sys, glob, shutil
import logging

from spice.core.bootstrapping.lib import *
from spice.core.optimal_tree.optimal_tree_ilp import *
from spice.core.multi_solutions.multi_solution_ilp import *
from spice.core.auxillary.ilp_functions import *

from spice.io.convert_input import *
from spice.io.spice_io import *
from spice.io.phasecns import *
from spice.postprocess.fill_gaps import *
from spice.postprocess.solution_probabilities import *

def run_spice_bootstrap(leaves_df, 
                        min_seg_size=0, 
                        max_n_segs=None,
                        keep_consec_segs=False,
                        allele_specific=True,
                        phase_alleles=False,
                        leaf_dependent_phasing=True,
                        max_cn=8,
                        verbose=True,
                        rs=1,
                        n_threads=1,
                        time_limit=5,
                        boot_distfunc='bphamm',
                        weight_wgd=True,
                        wgd_fract_genome=0.5,
                        multiple_solutions=True,
                        solution_pool_size=1000,
                        n_bootstraps=1000,
                        fixed_topology=False,
                        topology_prop_limit=0.025,
                        debug=False,
                        rank_multiple_trees=True,
                        output_type='save',
                        output_dir=None):
    
    save_output = False if output_type == 'return' else True
    if save_output:
        if output_dir is None:
            output_dir = './'
        elif (not os.path.exists(output_dir)):
            os.makedirs(output_dir)
    else:
        output_dir = None
    
    logger, logfile = initialize_logger(verbose, output_dir)
    
    logger.info('Processing input data')
    bootstrap_leaves, bootstrap_segsizes, leaves, leaf_name_mapping, all_leaves, final_mapping = process_spice_input(leaves_df, max_cn, phase_alleles, allele_specific, leaf_dependent_phasing, min_seg_size, max_n_segs, keep_consec_segs, logger)
    pos_seg_sizes = final_mapping.assign(segsize=final_mapping.End - final_mapping.Start).groupby(['Allele', 'Chr', 'finalPos'])['segsize'].sum().rename_axis(('Allele', 'Chr', 'Pos'))
    logger.info('Input data valid')
        
    run_start_time = datetime.datetime.now()
    time_limit = time_limit * 60 * 60
    
    if fixed_topology == False:
        logger.info('Bootstrapping topologies')
        res = tree_bootstrapping(n_bootstraps, bootstrap_leaves, bootstrap_segsizes, allele_specific, weight_wgd, max_cn, boot_distfunc, rs, njobs=n_threads)
        
        use_topologies = (res[['tree', 'topology_id']].value_counts()/n_bootstraps).reset_index().rename(columns={'count': 'proportion'}).sort_values(by='proportion', ascending=False)
        top_10_topologies = use_topologies.head(10)['topology_id'].values
        use_topologies = use_topologies[(use_topologies['proportion'] >= topology_prop_limit) | (use_topologies['topology_id'].isin(top_10_topologies))]['topology_id'].unique()

        if res.topology_id.nunique() - len(use_topologies) > 0:
            logger.warning('Only the top 10 topologies, or those with >={}% frequency, will be used in the ILP : removing {} topologies'.format(int(topology_prop_limit*100), len(res['topology_id'].unique()) - len(use_topologies)))
        
        res_use = res[res['topology_id'].isin(use_topologies)]
        unique_topologies = (res_use[['tree', 'topology_id']].value_counts()/len(res_use)).reset_index().rename(columns={'count': 'proportion'}).sort_values(by='proportion', ascending=False)
        unique_topologies['n_solutions'] = np.maximum(np.round(unique_topologies['proportion'] * solution_pool_size).astype(int), 1)
        unique_topologies['tree_tl'] = (time_limit/(len(unique_topologies)*2)) + (unique_topologies['proportion'] * (time_limit/2))
        unique_topologies['optimal_tl'] = (unique_topologies['tree_tl'] * 0.3).astype(int)
        unique_topologies['multiple_tl'] = (unique_topologies['tree_tl'] * 0.7).astype(int)

        logger.info('Bootstrapping complete : {} unique topologies'.format(len(unique_topologies)))
    else:
        assert type(fixed_topology) == tuple, 'Fixed topology must be a tuple of tuples representing the tree edges, e.g. (Parent, Child)'
        top_leaf_mapping = {v : k for k, v in leaf_name_mapping.items()}
        fixed_topology = [(top_leaf_mapping[i] if i in top_leaf_mapping else i for i in x) for x in fixed_topology]
        fixed_topology = tuple([tuple([int(str(i).replace('internal_', '')) for i in x]) for x in fixed_topology])
        unique_topologies = pd.DataFrame({'tree' : str(fixed_topology), 'topology_id' : 1, 'proportion' : 1.0, 'n_solutions' : solution_pool_size, 
                      'tree_tl' : time_limit, 'optimal_tl' : int(time_limit * 0.3), 'multiple_tl' : int(time_limit * 0.7)}, index=[0])
        unique_topologies['tree'] = unique_topologies['tree'].apply(lambda x: eval(x))

    logger.info(f'Running SPICE ILP: total time limit = {time_limit}s')
    
    total_opt_nodes = []
    total_multisol_nodes = []
    total_multisol_events = []
    total_opt_edges = []
    total_multisol_edges = []
    total_opt_solve_info = []
    total_multisol_solve_info = []
    total_opt_debug = []
    total_multisol_debug = []
    prop_missing_sols = 0
    ilp_start_time = datetime.datetime.now()
    
    for i, row in unique_topologies.iterrows():
        logger.info(f'Running SPICE ILP: topology {i+1} of {len(unique_topologies)} with {row.n_solutions} solutions : time limit = {row.tree_tl}s')
        optimal_nodes, optimal_edges, optimal_solve_info, optimal_model, optimal_debug = get_optimal_tree(leaves, 
                                                                                                        max_cn=max_cn, 
                                                                                                        verbose=verbose, 
                                                                                                        allele_specific=allele_specific, 
                                                                                                        tree_edges=row.tree, 
                                                                                                        time_limit=row.optimal_tl, 
                                                                                                        weight_wgd=weight_wgd,
                                                                                                        nthreads=n_threads, 
                                                                                                        mem=0.1,
                                                                                                        debug=debug,
                                                                                                        rs=rs,
                                                                                                        logger=logger,
                                                                                                        output_dir=output_dir)
        
        optimal_edges, optimal_model = check_wgd_presence(optimal_nodes, optimal_edges, optimal_model, pos_seg_sizes, leaf_name_mapping, wgd_fract_genome, logger)
        optimal_run_status = optimal_solve_info[optimal_solve_info.parameter == 'optimal_tree_model_status'].value.values[0]
        optimal_cost = optimal_solve_info[optimal_solve_info.parameter == 'optimal_tree_total_cost'].value.values[0]
        if optimal_run_status == 'optimal':
            logger.info(f'SPICE ILP completed to optimality for topology {i+1} of {len(unique_topologies)}')
        else:
            logger.warning(f'SPICE ILP did not complete to optimality for topology {i+1} of {len(unique_topologies)}: Model status = {optimal_run_status}')
            optimal_gap = optimal_solve_info[optimal_solve_info.parameter == 'optimal_tree_gap'].value.values[0]
            if optimal_gap > 0:
                logger.warning(f'Optimal solution not reached for topology {i+1} of {len(unique_topologies)}: Gap of {optimal_gap} found - consider increasing time limit')
        
        optimal_nodes['Solution'] = f'o{i+1}'       
        optimal_nodes['tree_id'] = i+1
        total_opt_nodes.append(optimal_nodes)
        
        optimal_edges['Solution'] = f'o{i+1}' 
        optimal_edges['tree_id'] = i+1
        total_opt_edges.append(optimal_edges)
        
        optimal_solve_info['tree_id'] = i+1
        total_opt_solve_info.append(optimal_solve_info)
        
        if optimal_debug is not None:
            optimal_debug.columns = [f'o{i+1}']
            total_opt_debug.append(optimal_debug)
            
        
        unique_topologies.loc[i, 'n_events'] = optimal_cost
            
        if multiple_solutions:
            multisol_nodes, multisol_events, multisol_edges, multisol_solve_info, multisol_debug = get_multiple_trees(leaves, 
                                                                                                    max_cn=max_cn, 
                                                                                                    verbose=verbose, 
                                                                                                    allele_specific=allele_specific, 
                                                                                                    tree_edges=row.tree, 
                                                                                                    time_limit=row.multiple_tl, 
                                                                                                    weight_wgd=weight_wgd,
                                                                                                    fix_wgd=True,
                                                                                                    solution_pool_size=row.n_solutions,
                                                                                                    init_model=optimal_model,
                                                                                                    sol_cutoff=optimal_cost*2, #change max gap from 2x optimal solution to parameter
                                                                                                    nthreads=n_threads, 
                                                                                                    mem=0.1,
                                                                                                    debug=debug,
                                                                                                    rs=rs,
                                                                                                    logger=logger,
                                                                                                    output_dir=output_dir)
            
            multisol_run_status = multisol_solve_info[multisol_solve_info.parameter == 'multi_solution_model_status'].value.values[0]    
            n_solutions = multisol_edges.Solution.nunique()
            if multisol_run_status == 'optimal':
                logger.info(f'Enumeration of solutions completed to optimality for topology {i+1} of {len(unique_topologies)}')
            else:
                multisol_gap = multisol_solve_info[multisol_solve_info.parameter == 'multi_solution_gap'].value.values[0]
                logger.warning(f'Enumeration of solutions did not complete to optimality for topology {i+1} of {len(unique_topologies)}: Model status = {multisol_run_status} - gap of {multisol_gap} found - consider increasing time limit')
                if n_solutions < row.n_solutions:
                    logger.warning(f'Only {n_solutions} of expected {row.n_solutions} solutions found for topology {i+1} of {len(unique_topologies)} - consider increasing time limit or reducing solution pool size')
                    missing_sols = 1 - (n_solutions / row.n_solutions)
                    prop_missing_sols = np.max([missing_sols, prop_missing_sols])       
            if ((n_solutions / row.n_solutions) < 1) & (1 - (n_solutions / row.n_solutions) == prop_missing_sols):
                logger.warning(f'{prop_missing_sols*100}% of solutions from previous topologies were not enumerated within the time limit, the number of solutions from other topologies will be adjusted to match. The total number of solutions will\
                            be reduced from {solution_pool_size} to {round(solution_pool_size*(1-prop_missing_sols))}') 
                    
            multisol_nodes['tree_id'] = i+1
            total_multisol_nodes.append(multisol_nodes)
            
            multisol_events['tree_id'] = i+1
            total_multisol_events.append(multisol_events)
            
            multisol_edges['tree_id'] = i+1
            total_multisol_edges.append(multisol_edges)
            
            multisol_solve_info['tree_id'] = i+1
            total_multisol_solve_info.append(multisol_solve_info)
            
            if multisol_debug is not None:
                multisol_debug['tree_id'] = i+1
                total_multisol_debug.append(multisol_debug)  
            
    if multiple_solutions:    
        total_multisol_nodes = concat_dfs(total_multisol_nodes, 1 - prop_missing_sols)
        total_nodes = pd.concat(total_opt_nodes + [total_multisol_nodes]).reset_index(drop=True)
        total_multisol_edges = concat_dfs(total_multisol_edges, 1 - prop_missing_sols)
        total_edges = pd.concat(total_opt_edges + [total_multisol_edges]).reset_index(drop=True)
        total_events = concat_dfs(total_multisol_events, 1 - prop_missing_sols) #! NEED TO ADD FILL GAPS EVENTS TO THIS!!
        total_solve_info = pd.concat(total_opt_solve_info + total_multisol_solve_info).reset_index(drop=True)
        if len(total_multisol_debug) > 0:
            total_multisol_debug = concat_dfs(total_multisol_debug, 1 - prop_missing_sols).pivot_table(index='index', columns='Solution', values='Value')
            total_debug = pd.concat(total_opt_debug + [total_multisol_debug], axis=1)
        else:
            total_debug = None
    else:
        min_event_tree = list(unique_topologies.sort_values(by='n_events', ascending=True).index)[0]
        total_nodes = total_opt_nodes[min_event_tree]
        total_edges = total_opt_edges[min_event_tree]
        total_solve_info = total_opt_solve_info[min_event_tree]
        total_events = None
        total_debug = total_opt_debug[min_event_tree]
        
    solution_topology_mapping = total_edges[['Solution', 'tree_id']].drop_duplicates()
    solution_topology_mapping = dict(zip(solution_topology_mapping.Solution, solution_topology_mapping.tree_id))

    ilp_end_time = datetime.datetime.now()
    total_ilp_time = (ilp_end_time - ilp_start_time).total_seconds()
    logger.info(f'SPICE ILP complete: total ILP time = {np.round(total_ilp_time, 2)}')

    if len(all_leaves[all_leaves.filter_seg]) > 0:
        logger.info('Filling CN small segs')
        all_leaves['Node'] = all_leaves['Node'].map({v : k for k, v in leaf_name_mapping.items()})
        total_nodes, total_edges = fill_removed_segments(total_nodes, total_edges, all_leaves, final_mapping, max_cn, n_threads, logger)
        logger.info('Filling CN small segs complete')
    else:
        final_mapping['Allele'] = final_mapping['Allele'].map({'A' : 0, 'B' : 1})
        total_nodes = total_nodes.merge(final_mapping, left_on=['Chr', 'Allele', 'Pos'], 
                                                right_on=['Chr', 'Allele', 'finalPos'], how='left').drop(['finalPos', 'Pos'], axis=1).rename(columns={'origPos' : 'Pos'})
        total_nodes = convert_long_to_wide(total_nodes.assign(method='spice'))
    
    col_order = ['tree_id', 'Solution', 'Node', 'Chr', 'Start', 'End', 'startPos', 'endPos', 'Allele', 'Event', 'nEvents']
    startPos_mapping = final_mapping.groupby(['Chr', 'Allele', 'finalPos'])[['origPos', 'Start']].min().reset_index()
    endPos_mapping = final_mapping.groupby(['Chr', 'Allele', 'finalPos'])[['origPos', 'End']].max().reset_index()
    
    if total_events is not None:
        total_events = total_events.rename(columns={'startPos' : 'finalPos'}) \
                            .merge(startPos_mapping, on=['Chr', 'Allele', 'finalPos'], how='left') \
                            .drop(['finalPos'], axis=1) \
                            .rename(columns={'origPos' : 'startPos', 'endPos' : 'finalPos'}) \
                            .merge(endPos_mapping, on=['Chr', 'Allele', 'finalPos'], how='left') \
                            .drop(['finalPos'], axis=1) \
                            .rename(columns={'origPos' : 'endPos'})[col_order]

    mapping = {k : leaf_name_mapping[k] if k in leaf_name_mapping.keys() else 'internal_' + str(k) for k in total_nodes['Node'].unique()}

    total_nodes['Node'] = total_nodes['Node'].map(mapping, na_action='ignore')
    total_nodes['tree_id'] = total_nodes['Solution'].map(solution_topology_mapping)
    total_edges['Parent'] = total_edges['Parent'].map(mapping, na_action='ignore')
    total_edges['Child'] = total_edges['Child'].map(mapping, na_action='ignore')
    total_edges['tree_id'] = total_edges['Solution'].map(solution_topology_mapping)
    if total_events is not None:
        total_events['Node'] = total_events['Node'].map(mapping, na_action='ignore')
        total_events['tree_id'] = total_events['Solution'].map(solution_topology_mapping)

    if multiple_solutions:
        total_nodes, total_edges, total_events = check_for_duplicates(total_nodes, total_edges, total_events, logger)
        
    # NEW - SPLIT INTO OPTIMAL AND MULTIPLE SOLUTIONS
    total_nodes_optimal = total_nodes[total_nodes.Solution.str.contains('o')]
    total_nodes_multisol = total_nodes[total_nodes.Solution.str.contains('m')].copy()
    total_edges_optimal = total_edges[total_edges.Solution.str.contains('o')]
    total_edges_multisol = total_edges[total_edges.Solution.str.contains('m')].copy()
        
    if save_output:
        total_nodes_optimal.to_csv(os.path.join(output_dir, 'spice_total_nodes_optimal.tsv.gz'), sep='\t', index=False)
        total_nodes_multisol.to_csv(os.path.join(output_dir, 'spice_total_nodes.tsv.gz'), sep='\t', index=False)
        total_edges_optimal.to_csv(os.path.join(output_dir, 'spice_total_edges_optimal.tsv.gz'), sep='\t', index=False)
        total_edges_multisol.to_csv(os.path.join(output_dir, 'spice_total_edges.tsv.gz'), sep='\t', index=False)
        if total_events is not None:
            total_events.to_csv(os.path.join(output_dir, 'spice_total_events.tsv.gz'), sep='\t', index=False)
        if total_debug is not None:
            total_debug.to_csv(os.path.join(output_dir, 'spice_debug.tsv.gz'), sep='\t', index=False)
            
    solution_probabilities = None
    chosen_topology_cn_probabilities = None
    if rank_multiple_trees == True:
        if multiple_solutions == False:
            logger.warning('Multiple solutions parameter set to False, no multiple trees to rank')
        else:
            for weight_func in ['exp_min', 'geometric']:
                logger.info(f'Ranking multiple solutions using {weight_func} weighting function')
                solution_probabilities, chosen_topology_cn_probabilities = get_solution_probabilities(total_edges_multisol, total_nodes_multisol, logger, gap=0, weight_func=weight_func) 
                if save_output:
                    save_ms_suff = '_' + weight_func.replace('_', '')
                    solution_probabilities.to_csv(os.path.join(output_dir, f'spice_solutions_ranked{save_ms_suff}.tsv.gz'), sep='\t', index=False)
                    chosen_topology_cn_probabilities.to_csv(os.path.join(output_dir, f'spice_cn_probabilities{save_ms_suff}.tsv.gz'), sep='\t', index=False)
            logger.info('Ranking multiple solutions complete')   
            
    run_time = (datetime.datetime.now() - run_start_time).total_seconds()
            
    run_info = pd.DataFrame({'total_run_time' : run_time, 'random_seed' : rs}, index=['value']).T.reset_index().rename(columns={'index':'parameter'})
    run_info = pd.concat([total_solve_info, run_info]).reset_index(drop=True)
    if save_output:
        run_info.to_csv(os.path.join(output_dir, 'spice_solve_info.tsv'), sep='\t', index=False) 
           
    if save_output:
        for node_dir in glob.glob(output_dir + '/grbnodes*/'):
            shutil.rmtree(os.path.join(node_dir))
                    
    logger.info('SPICE complete')
    logger.handlers.clear()

    if output_type != 'save':
        results = {'total_nodes_optimal' : total_nodes_optimal, 'total_edges_optimal' : total_edges_optimal, 
                   'total_edges_multisol' : total_edges_multisol, 'total_nodes_multisol' : total_nodes_multisol,
                   'total_events' : total_events, 'solution_probabilities' : solution_probabilities, 
                   'chosen_topology_cn_probabilities' : chosen_topology_cn_probabilities, 
                   'total_debug' : total_debug, 'solve_info' : run_info, 'mapping': final_mapping}
        return {k : v for k, v in results.items() if v is not None} 
    
    
def check_for_duplicates(total_nodes, total_edges, total_events, logger):
    comp_te = total_edges.groupby('Solution').apply(lambda x: tuple(map(tuple, x.values)), include_groups=False)
    comp_tn = total_nodes.sort_values(by=['Solution', 'Node', 'Chr', 'Pos']).groupby('Solution')[['cnA', 'cnB']].apply((lambda x: tuple(map(tuple, x.values)))).sort_values()
    comp_tev = total_events.groupby('Solution').apply(lambda x: tuple(map(tuple, x.values)), include_groups=False)
    dups = pd.concat([comp_te, comp_tn, comp_tev], axis=1).reset_index().groupby([0, 1, 2])['Solution'].agg(Solution=list, n='count').reset_index()[['Solution', 'n']].sort_values(by='n', ascending=False)
    if dups['n'].max() == 1:
        logger.info(f'No duplicate solutions found')
        return total_nodes, total_edges, total_events
    else:
        logger.warning(f'Found {dups[dups.n > 1]["n"].sum()} duplicate solutions')
        dups['dup_idx'] = np.where(dups['n'] > 1, np.char.add('idx_', np.arange(1, len(dups)+1).astype(str)), np.nan)
        dup_mapping = dups.explode('Solution').set_index('Solution')['dup_idx'].to_dict()
        total_nodes['dup_idx'] = total_nodes['Solution'].map(dup_mapping)
        total_edges['dup_idx'] = total_edges['Solution'].map(dup_mapping)
        total_events['dup_idx'] = total_events['Solution'].map(dup_mapping)
        return total_nodes, total_edges, total_events
    
    
def concat_dfs(solution_list, prop_sols_used):
    processed_list = []
    solution_count = 0
    for i in solution_list:
        processed_sols, solution_count = process_output(i, prop_sols_used, solution_count)
        processed_list.append(processed_sols)
    return pd.concat(processed_list).reset_index(drop=True)


def process_output(solutions, prop_sols_used, solution_count=0):
    max_sol_n = np.maximum(round(solutions['Solution'].nunique() * prop_sols_used), 1)
    solutions = solutions[solutions.Solution <= max_sol_n]
    solutions['Solution'] = 'm' + (solutions['Solution'] + solution_count).astype(str)
    solution_count += max_sol_n 
    return solutions, solution_count


def convert_long_to_wide(total_nodes):
    total_nodes['Allele'] = total_nodes['Allele'].map({0 : 'A', 1 : 'B'})
    cn = total_nodes.rename(columns={'CN' : 'cn'}).pivot_table(index=['tree_id', 'Solution', 'Node', 'Chr', 'Pos', 'Start', 'End'], columns='Allele', values=['cn', 'Gains', 'Deletions'])
    cn.columns = [''.join(col) for col in cn.columns.values]
    return cn.reset_index()

def cn_dict_to_df(cn_dict):
    cn_df = pd.DataFrame([{'Node': node, 'Chr' : ch, 'Allele' : ['A', 'B'][allele], 'Pos' : pos+1, 'CN' : cn} for node in cn_dict
                                                                                                for ch in cn_dict[node]
                                                                                                for allele in range(len(cn_dict[node][ch]))
                                                                                                for pos, cn in enumerate(cn_dict[node][ch][allele])])
    return cn_df

    