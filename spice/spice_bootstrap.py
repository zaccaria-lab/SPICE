import pyomo.environ as pe
import numpy as np
import datetime
import os, sys, glob, shutil
import logging
import tracemalloc

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
                        min_seg_size=5e4, 
                        max_n_segs=None,
                        keep_consec_segs=False,
                        allele_specific=True,
                        phase_alleles=False,
                        leaf_dependent_phasing=True,
                        max_cn=8,
                        focal_max_cn=None,
                        verbose=True,
                        rs=1,
                        n_threads=1,
                        time_limit=5,
                        weight_wgd=True,
                        wgd_fract_genome=0.5,
                        multiple_solutions=True,
                        solution_pool_size=1000,
                        max_cutoff=2, 
                        n_bootstraps=1000,
                        fixed_topology=False,
                        topology_prop_limit=0.025,
                        topology_topn=10,
                        initnodes=None,
                        debug=False,
                        rm_small_segs=False,
                        rank_multiple_trees=True,
                        output_type='save',
                        output_dir=None):
    
    '''Main function to run SPICE with bootstrapping of input data and ILP 
    topologies.
    
    Parameters:
    -----------
    leaves_df : pd.DataFrame
        DataFrame containing the leaf copy number profiles, with columns 'Node', 'Chr', 'Start', 'End', 'Allele', 'CN'.
    min_seg_size : int, optional
        Minimum segment size to consider in the ILP (interval model), any segments smaller than this are reconstructed 
        using the alternative small segments model (default: 0).
    max_n_segs : int, optional
        Maximum number of segments to consider in the ILP (interval model), the largest N segments are inferred using 
        the ILP model and the remaining segments are reconstructed using the alternative small segments model (default: 
        None, i.e. no limit).
    keep_consec_segs : bool, optional
        Whether to prevent filtering of consecutive segments according to min_seg_size or max_n_segs, this will increase
        the number of segments in the ILP model (default: False, i.e. consecutive segments can be filtered for the small 
        segments model).
    allele_specific : bool, optional
        Whether to run SPICE in allele-specific mode, where each allele is considered separately in the ILP model 
        (default: True).
    phase_alleles : bool, optional
        Whether to phase alleles prior to running SPICE, this will attempt to phase alleles based on the input copy 
        number profiles using the phasingalgorithm from CHISEL (Zaccaria & Raphael, Nat Biotechnol. 2021) (default: 
        False).
    leaf_dependent_phasing : bool, optional
        Whether to prevent the independent phasing of copy number segments in different input leaf profiles. (default: 
        True, i.e. phasing of a copy number segment occurs in all leaf nodes).
    max_cn : int, optional
        Maximum copy number to consider in the ILP model (default: 8).
    focal_max_cn : int, optional
        Maximum copy number to consider for the small segments model (default:  None, i.e. same max_cn as ILP model).
    verbose : bool, optional
        Whether to print verbose output during the SPICE run (default: True).
    rs : int, optional
        Random seed to use for bootstrapping and ILP solver (default: 1).
    n_threads : int, optional
        Number of threads to use for ILP solver (default: 1).
    time_limit : int, optional
        Time limit in hours for the entire SPICE run (default: 5).
    weight_wgd : bool, optional
        Whether to apply a weight to WGD events (equivalent to the highest possible CNA amplitude, i.e. the max_cn value) 
        in the ILP objective function (default: True).
    wgd_fract_genome : float, optional
        The minimum fraction of each allele (if allele_specific==True), or the genome, that must be affected by WGD 
        events for a WGD to be called present in the ILP model (default: 0.5).
    multiple_solutions : bool, optional
        Whether to enumerate multiple solutions (default: True).
    solution_pool_size : int, optional
        The number of solutions to enumerate if multiple_solutions is True (default: 1000).
    max_cutoff : float, optional
        The maximum cost cutoff for enumerating solutions in the multiple solutions ILP, as a multiple of the optimal 
        cost (default: 2, i.e. only solutions with cost <= 2x the optimal cost will be enumerated).
    n_bootstraps : int, optional
        The number of bootstraps to run when inferring the phylogenetic tree topology. This parameter is ignored if a 
        fixed topology is provided (default: 1000).
    fixed_topology : tuple of tuples, optional
        A fixed topology for the inferred phylogenetic tree, represented as a tuple of tuples (Parent, Child) 
        representing the tree edges. If provided, no inference of topologies will be performed and only this topology 
        will be used in the ILP model (default: False, i.e. topologies are inferred).
    topology_prop_limit : float, optional
        The proportion of bootstrapped trees that must have a given topology for that topology to be included in the ILP
        model (default: 0.025, i.e. topologies must be present in at least 2.5% of bootstrapped trees to be included).
    initnodes : pd.DataFrame, optional
        DataFrame containing initial node copy number profiles to use for warm-starting the ILP solver (i.e. from 
        MEDICC2), with columns 'Node', 'Chr', 'Start', 'End', 'Allele', 'CN'. Note: segments must be defined the same in
        the leaves_df and initnodes Dataframes. If initnodes is provided, fixed_topology must be provided with the 
        corresponding topology (default: None, i.e. no initial node profiles provided). 
    debug : bool, optional
        Whether SPICE is being run in debug mode, which outputs all variables from the ILP model (default: False).
    rm_small_segs: bool, optional
        Whether to filter out small segments from the data (where # small segments are defined using the min_seg_size or 
        max_n_segs parameters), otherwise the evolution of these segments are inferred using the small segments model 
        (default: False).
    rank_multiple_trees: bool, optional
        Whether to calculate phylogeny probabilities and rank alternative phylogenies (default: True).    
    output_type : str, optional
        Whether to save or return the output. Options: save, return, both (default: save).
    output_dir : {'save', 'return', 'both}, optional
        The directory in which to save the output of SPICE. Only used if output_type != return (default: None).
    
   Returns
   -----------
    dict or None
        A filtered dictionary of SPICE dataframes (None-valued keys are excluded).
        Only returned when ``output_type != 'save'``. Keys include:

        - total_nodes_optimal : Inferred copy number profiles output by the optimal ILP model (for each topology inferred 
        by SPICE), with columns 'Solution', 'Node', 'Chr', 'Pos', 'Start', 'End', 'cnA', 'cnB', 'DeletionsA', 'DeletionsB'
        'GainsA', 'GainsB', 'method'. 
        - total_nodes_multisol : Inferred copy number profiles output by the enumerate ILP model (for each topology 
        inferred by SPICE), with columns 'Solution', 'Node', 'Chr', 'Pos', 'Start', 'End', 'cnA', 'cnB', 'DeletionsA', 
        'DeletionsB', 'GainsA', 'GainsB', 'method'.   
        - total_edges_optimal : Inferred topology information output by SPICE for each edge in the phylogeny inferred 
        using the optimal ILP model, with columns 'Parent', 'Child', 'Events', 'Deletions', 'Gains', 'WGD', 'Total_cost',
        'Gurobi_cost', 'tree_id', 'Total_cost_focal', 'Total_cost_focal_interval'.
        - total_edges_multisol : Inferred topology information output by SPICE for each edge in each phylogeny inferred 
        using the enumerate ILP model, with columns 'Parent', 'Child', 'Events', 'Deletions', 'Gains', 'WGD',
        'Total_cost', 'Gurobi_cost', 'tree_id', 'Total_cost_focal', 'Total_cost_focal_interval'.
        - total_events : The inferred copy number events from the enumerate ILP for all solutions, with columns 'tree_id',
        'Solution', 'Node', 'Chr', 'Start', 'End', 'startPos', 'endPos', 'Allele', 'Event', 'nEvents'.
        - solution_probabilities : The probability of each phylogeny inferred in SPICE and the subsequent ranking.
        - chosen_topology_cn_probabilities : The probability of each copy number state of ancestral nodes, calculated 
        from copy number profiles of solutions with the highest-ranked topology.
        - total_debug : debug information.
        - solve_info : run/solver metadata and info.
        - mapping : mapping information for genomic segments, containing 'origPos' (input copy number segments), 
        'mergePos' (segments after collapsing consecutive segments with equal copy numbers), 'filteredPos' (NaN for any 
        segment inferred using the small segments model), and 'finalPos' (the genomic segments input to each ILP model).
    '''
    
    save_output = False if output_type == 'return' else True
    if save_output:
        if output_dir is None:
            output_dir = './'
        elif (not os.path.exists(output_dir)):
            os.makedirs(output_dir)
    else:
        output_dir = None
    
    logger, logfile = initialize_logger(verbose, output_dir)
    
    # starting memory monitoring
    tracemalloc.start() 
    
    # starting time monitoring
    run_start_time = datetime.datetime.now()
    time_limit = time_limit * 60 * 60
    
    # PROCESS INPUT DATA
    logger.info('Processing input data')
    if (fixed_topology == False) and (leaves_df['Node'].dtype == int):
        leaves_df['Node'] = 'clone_' + leaves_df['Node'].astype(str)
    (bootstrap_leaves, bootstrap_segsizes, leaves, 
    node_name_mapping, all_leaves, pos_mapping) = process_spice_input(leaves_df, max_cn, phase_alleles, allele_specific, 
                                                                      leaf_dependent_phasing, min_seg_size, max_n_segs, 
                                                                      keep_consec_segs, focal_max_cn, fixed_topology, 
                                                                      logger)
    pos_seg_sizes = (pos_mapping
                    .assign(segsize=pos_mapping.End - pos_mapping.Start)
                    .groupby(['Allele', 'Chr', 'finalPos'])['segsize']
                    .sum()
                    .rename_axis(('Allele', 'Chr', 'Pos')))
    logger.info('Input data valid')
    
    # RUN BOOTSTRAPPING
    if fixed_topology == False:
        logger.info('Bootstrapping topologies')
        res = tree_bootstrapping(n_bootstraps, bootstrap_leaves, bootstrap_segsizes, allele_specific, weight_wgd, 
                                max_cn, 'bphamm', rs, njobs=n_threads)
        use_topologies = ((res[['tree', 'topology_id']].value_counts()/n_bootstraps)
                          .reset_index()
                          .rename(columns={'count': 'proportion'})
                          .sort_values(by='proportion', ascending=False))
        top_10_topologies = use_topologies.head(topology_topn)['topology_id'].values
        use_topologies = (use_topologies[(use_topologies['proportion'] >= topology_prop_limit) | 
                                         (use_topologies['topology_id'].isin(top_10_topologies))]
                          ['topology_id'].unique())

        if res.topology_id.nunique() - len(use_topologies) > 0:
            logger.warning('Only the top {} topologies, plus those with >={}% frequency, will be used in the ILP : removing {} topologies'.format(topology_topn, int(topology_prop_limit*100), 
                                                                                    len(res['topology_id'].unique()) - 
                                                                                    len(use_topologies)))
        
        res_use = res[res['topology_id'].isin(use_topologies)]
        unique_topologies = ((res_use[['tree', 'topology_id']].value_counts()/len(res_use))
                             .reset_index()
                             .rename(columns={'count': 'proportion'})
                             .sort_values(by='proportion', ascending=False))
        unique_topologies['n_solutions'] = np.maximum(np.round(unique_topologies['proportion'] * solution_pool_size).astype(int), 1)
        unique_topologies['tree_tl'] = (time_limit/(len(unique_topologies)*2)) + (unique_topologies['proportion'] * (time_limit/2))
        unique_topologies['optimal_tl'] = (unique_topologies['tree_tl'] * 0.3).astype(int)
        unique_topologies['multiple_tl'] = (unique_topologies['tree_tl'] * 0.7).astype(int)
        logger.info('Bootstrapping complete : {} unique topologies'.format(len(unique_topologies)))
    else:
        assert type(fixed_topology) == tuple, 'Fixed topology must be a tuple of tuples representing the tree edges, e.g. (Parent, Child)'
        fixed_topology = [(node_name_mapping[i], node_name_mapping[j]) for i, j in fixed_topology]
        unique_topologies = pd.DataFrame({'tree' : str(fixed_topology), 'topology_id' : 1, 'proportion' : 1.0, 
                                          'n_solutions' : solution_pool_size, 'tree_tl' : time_limit, 
                                          'optimal_tl' : int(time_limit * 0.3), 'multiple_tl' : int(time_limit * 0.7)}, 
                                         index=[0])
        unique_topologies['tree'] = unique_topologies['tree'].apply(lambda x: eval(x))
        
    if save_output:
        mapping = {v : k for k, v in node_name_mapping.items()}
        save_topologies = unique_topologies.copy().explode('tree')
        save_topologies['Parent'], save_topologies['Child'] = zip(*save_topologies['tree'])
        save_topologies['Parent'] = save_topologies['Parent'].map(mapping, na_action='ignore').astype(str)
        save_topologies['Child'] = save_topologies['Child'].map(mapping, na_action='ignore').astype(str)
        save_topologies.to_csv(os.path.join(output_dir, 'spice_topologies.tsv.gz'), sep='\t', index=False)

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
    
    if initnodes is not None:
        assert fixed_topology != False, 'If initialising internal copy number profiles, the topology must be fixed.'
        opt_init_dict = process_medicc_initialise(initnodes, pos_mapping, node_name_mapping)
        logger.info('CN states provided to initialise optimal ILP, running with warmstart.')
    else:
        opt_init_dict = None
    
    # RUN OPTIMAL ILP
    for i, row in unique_topologies.iterrows():
        logger.info(f'Running SPICE ILP: topology {i+1} of {len(unique_topologies)} with {row.n_solutions} solutions : time limit = {row.tree_tl}s')
        (optimal_nodes, optimal_edges, optimal_solve_info, 
         optimal_model, optimal_debug) = get_optimal_tree(leaves, max_cn=max_cn, verbose=verbose, 
                                                          allele_specific=allele_specific, tree_edges=row.tree, 
                                                          time_limit=row.optimal_tl, weight_wgd=weight_wgd,
                                                          init_model=opt_init_dict, nthreads=n_threads, mem=0.1,
                                                          debug=debug, rs=rs, logger=logger, output_dir=output_dir)
        
        optimal_edges, optimal_model = check_wgd_presence(optimal_nodes, optimal_edges, optimal_model, pos_seg_sizes, 
                                                          node_name_mapping, wgd_fract_genome, logger)
        optimal_run_status = optimal_solve_info[optimal_solve_info.parameter == 'optimal_tree_model_status'].value.values[0]
        optimal_cost = optimal_solve_info[optimal_solve_info.parameter == 'optimal_tree_total_cost'].value.values[0]
        if optimal_run_status == 'optimal':
            logger.info(f'SPICE ILP completed to optimality for topology {i+1} of {len(unique_topologies)}')
        else:
            logger.warning(f'SPICE ILP did not complete to optimality for topology {i+1} of {len(unique_topologies)}: status = {optimal_run_status}')
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
        
        # RUN ENUMERATE SOLUTIONS ILP    
        if multiple_solutions:
            (multisol_nodes, multisol_events, multisol_edges, 
            multisol_solve_info, multisol_debug) = get_multiple_trees(leaves, max_cn=max_cn, verbose=verbose, 
                                                                      allele_specific=allele_specific, 
                                                                      tree_edges=row.tree, time_limit=row.multiple_tl, 
                                                                      weight_wgd=weight_wgd, fix_wgd=True,
                                                                      solution_pool_size=row.n_solutions,
                                                                      init_model=optimal_model, sol_cutoff=optimal_cost*max_cutoff,
                                                                      nthreads=n_threads, mem=0.1, debug=debug, rs=rs,
                                                                      logger=logger, output_dir=output_dir)
            
            multisol_run_status = (multisol_solve_info[multisol_solve_info.parameter == 'multi_solution_model_status']
                                   .value
                                   .values[0])  
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
                logger.warning(f'{prop_missing_sols*100}% of solutions from previous topologies were not enumerated within the time limit, the number of solutions from other topologies will be adjusted to match. The total number of solutions will be reduced from {solution_pool_size} to {round(solution_pool_size*(1-prop_missing_sols))}') 
                    
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
    
    # PROCESS ALL OUTPUT        
    if multiple_solutions:    
        total_multisol_nodes = concat_dfs(total_multisol_nodes, 1 - prop_missing_sols)
        total_nodes = pd.concat(total_opt_nodes + [total_multisol_nodes]).reset_index(drop=True)
        total_multisol_edges = concat_dfs(total_multisol_edges, 1 - prop_missing_sols)
        total_edges = pd.concat(total_opt_edges + [total_multisol_edges]).reset_index(drop=True)
        total_events = concat_dfs(total_multisol_events, 1 - prop_missing_sols) #! NEED TO ADD FILL GAPS EVENTS TO THIS!!
        total_solve_info = pd.concat(total_opt_solve_info + total_multisol_solve_info).reset_index(drop=True)
        if len(total_multisol_debug) > 0:
            total_multisol_debug = (concat_dfs(total_multisol_debug, 1 - prop_missing_sols)
                                    .pivot_table(index='index', columns='Solution', values='Value'))
            total_debug = pd.concat(total_opt_debug + [total_multisol_debug], axis=1)
        else:
            total_debug = None
    else:
        min_event_tree = list(unique_topologies.sort_values(by='n_events', ascending=True).index)[0]
        total_nodes = total_opt_nodes[min_event_tree]
        total_edges = total_opt_edges[min_event_tree]
        total_solve_info = total_opt_solve_info[min_event_tree]
        total_events = None
        if debug:
            total_debug = total_opt_debug[min_event_tree]
        else:
            total_debug = None
        
    solution_topology_mapping = total_edges[['Solution', 'tree_id']].drop_duplicates()
    solution_topology_mapping = dict(zip(solution_topology_mapping.Solution, solution_topology_mapping.tree_id))

    ilp_end_time = datetime.datetime.now()
    total_ilp_time = (ilp_end_time - ilp_start_time).total_seconds()
    logger.info(f'SPICE ILP complete: total ILP time = {np.round(total_ilp_time, 2)}')
    
    mapping = {v : k for k, v in node_name_mapping.items()}
    total_nodes['Node'] = total_nodes['Node'].map(mapping, na_action='ignore').astype(str)
    total_nodes['tree_id'] = total_nodes['Solution'].map(solution_topology_mapping)
    total_edges['Parent'] = total_edges['Parent'].map(mapping, na_action='ignore').astype(str)
    total_edges['Child'] = total_edges['Child'].map(mapping, na_action='ignore').astype(str)
    total_edges['tree_id'] = total_edges['Solution'].map(solution_topology_mapping)
    if total_events is not None:
        total_events['Node'] = total_events['Node'].map(mapping, na_action='ignore').astype(str)
        total_events['tree_id'] = total_events['Solution'].map(solution_topology_mapping)

    # RUN SMALL SEGMENTS INFERENCE
    if (len(all_leaves[all_leaves.filter_seg]) > 0) and (rm_small_segs == False):
        logger.info('Filling CN small segs')
        all_leaves['Node'] = all_leaves['Node'].astype(str)
        total_nodes, total_edges = fill_removed_segments(total_nodes, total_edges, all_leaves, pos_mapping, max_cn, 
                                                         n_threads, focal_maxcn=focal_max_cn, logger=logger, 
                                                         allele_specific=allele_specific)
        logger.info('Filling CN small segs complete')
    else:
        pos_mapping['Allele'] = pos_mapping['Allele'].map({'A' : 0, 'B' : 1})
        total_nodes = (total_nodes
                       .merge(pos_mapping, left_on=['Chr', 'Allele', 'Pos'], right_on=['Chr', 'Allele', 'finalPos'], how='left')
                       .drop(['finalPos', 'Pos'], axis=1)
                       .rename(columns={'origPos' : 'Pos'}))
        total_nodes = convert_long_to_wide(total_nodes.assign(method='spice'))
    
    # CONVERT SPICE RUN GENOME POSITIONS BACK TO ORIGINAL POSITIONS
    col_order = ['tree_id', 'Solution', 'Node', 'Chr', 'Start', 'End', 'startPos', 'endPos', 'Allele', 'Event', 'nEvents']
    startPos_mapping = pos_mapping.groupby(['Chr', 'Allele', 'finalPos'])[['origPos', 'Start']].min().reset_index()
    endPos_mapping = pos_mapping.groupby(['Chr', 'Allele', 'finalPos'])[['origPos', 'End']].max().reset_index()
    
    if total_events is not None:
        total_events = (total_events
                        .rename(columns={'startPos' : 'finalPos'}) 
                        .merge(startPos_mapping, on=['Chr', 'Allele', 'finalPos'], how='left') 
                        .drop(['finalPos'], axis=1) 
                        .rename(columns={'origPos' : 'startPos', 'endPos' : 'finalPos'}) 
                        .merge(endPos_mapping, on=['Chr', 'Allele', 'finalPos'], how='left') 
                        .drop(['finalPos'], axis=1) 
                        .rename(columns={'origPos' : 'endPos'})[col_order])


        
        
    # ENSURE THAT EVERY SOLUTION IS UNIQUE
    if multiple_solutions:
        total_nodes, total_edges, total_events = check_for_duplicates(total_nodes, total_edges, total_events, logger)
        
    # SPLIT INTO OPTIMAL AND MULTIPLE SOLUTIONS
    total_nodes_optimal = total_nodes[total_nodes.Solution.str.contains('o')]
    total_nodes_multisol = total_nodes[total_nodes.Solution.str.contains('m')].copy()
    total_edges_optimal = total_edges[total_edges.Solution.str.contains('o')]
    total_edges_multisol = total_edges[total_edges.Solution.str.contains('m')].copy()
    
    # SAVE SPICE OUTPUT    
    if save_output:
        total_nodes_optimal.to_csv(os.path.join(output_dir, 'spice_total_nodes_optimal.tsv.gz'), sep='\t', index=False)
        total_nodes_multisol.to_csv(os.path.join(output_dir, 'spice_total_nodes.tsv.gz'), sep='\t', index=False)
        total_edges_optimal.to_csv(os.path.join(output_dir, 'spice_total_edges_optimal.tsv.gz'), sep='\t', index=False)
        total_edges_multisol.to_csv(os.path.join(output_dir, 'spice_total_edges.tsv.gz'), sep='\t', index=False)
        if total_events is not None:
            total_events.to_csv(os.path.join(output_dir, 'spice_total_events.tsv.gz'), sep='\t', index=False)
        if total_debug is not None:
            total_debug.to_csv(os.path.join(output_dir, 'spice_debug.tsv.gz'), sep='\t', index=False)
    
    # RANK MULTIPLE SOLUTIONS        
    solution_probabilities = None
    chosen_topology_cn_probabilities = None
    if rank_multiple_trees == True:
        if multiple_solutions == False:
            logger.warning('Multiple solutions parameter set to False, no multiple trees to rank')
        else:
            weight_func = 'geometric'
            logger.info(f'Ranking multiple solutions using {weight_func} weighting function')
            (solution_probabilities, 
            chosen_topology_cn_probabilities) = get_solution_probabilities(total_edges_multisol, total_nodes_multisol, 
                                                                           logger, gap=0, weight_func=weight_func, 
                                                                           allele_specific=allele_specific) 
            if save_output:
                solution_probabilities.to_csv(os.path.join(output_dir, f'spice_solutions_ranked.tsv.gz'), sep='\t', index=False)
                chosen_topology_cn_probabilities.to_csv(os.path.join(output_dir, f'spice_cn_probabilities.tsv.gz'), sep='\t', index=False) 
            logger.info('Ranking multiple solutions complete')   
    
    # CALCULATE RUN INFORMATION        
    run_time = (datetime.datetime.now() - run_start_time).total_seconds()
    logger.info(f'Total SPICE run time: {np.round(run_time, 2)}s')
    
    # CALCULATE MEMORY INFORMATION
    peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024) # Convert to MB
    tracemalloc.stop() 
    logger.info(f'Peak memory usage: {np.round(peak_mem, 2)} MB')
            
    run_info = (pd.DataFrame({'total_run_time' : run_time, 'random_seed' : rs}, index=['value'])
                .T
                .reset_index()
                .rename(columns={'index':'parameter'}))
    run_info = pd.concat([total_solve_info, run_info]).reset_index(drop=True)
    if save_output:
        run_info.to_csv(os.path.join(output_dir, 'spice_solve_info.tsv'), sep='\t', index=False) 
           
    if save_output:
        for node_dir in glob.glob(output_dir + '/grbnodes*/'):
            shutil.rmtree(os.path.join(node_dir))
                    
    logger.info('SPICE complete')
    logger.handlers.clear()

    # RETURN SPICE OUTPUTS
    if output_type != 'save':
        results = {'total_nodes_optimal' : total_nodes_optimal, 'total_edges_optimal' : total_edges_optimal, 
                   'total_edges_multisol' : total_edges_multisol, 'total_nodes_multisol' : total_nodes_multisol,
                   'total_events' : total_events, 'solution_probabilities' : solution_probabilities, 
                   'chosen_topology_cn_probabilities' : chosen_topology_cn_probabilities, 
                   'total_debug' : total_debug, 'solve_info' : run_info, 'mapping': pos_mapping}
        return {k : v for k, v in results.items() if v is not None} 
    
    


    