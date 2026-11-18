"""Run SPICE from start to finish: infer phylogenies with ancestral copy number profiles from leaf copy number profiles.

`run_spice_bootstrap` is the entry point, called by the command-line tool (`spice/bin/spice.py`). It:

1. Processes the input: optionally phases the alleles, merges consecutive bins with the same copy number in every
   leaf into segments, and flags small segments for the focal evolutionary model (`spice.libs.preprocess`).
2. Finds candidate tree topologies by bootstrapping over chromosomes (`spice.libs.topology.bootstrap`), unless a fixed
   topology is given.
3. For each topology, solves the optimal ILP (`spice.libs.ilp.optimal`) and, optionally, enumerates further solutions
   with the multiple-solutions ILP (`spice.libs.ilp.multi`). The time limit is shared between the topologies.
4. Fills in the copy numbers of the small segments left out of the ILP (`spice.libs.postprocess.fill_gaps`), unless
   `rm_small_segs` is True.
5. Ranks the multiple solutions and gives copy number probabilities for each topology
   (`spice.libs.postprocess.solution_probabilities`).
6. Returns the result tables, and saves each one to `output_dir` as soon as it is final (see `output_type`).
"""

import datetime
import os
import glob
import shutil
import tracemalloc
from typing import Optional, Union

import numpy as np
import pandas as pd

from spice.libs.utils.logging import close_handlers, initialize_logger
from spice.libs.utils.conversion import expand_segments, map_to_original_positions, sort_by_columns
from spice.libs.preprocess.input import process_spice_input, standardise_alleles
from spice.libs.topology.bootstrap import tree_bootstrapping
from spice.libs.ilp.optimal.solve_model import get_optimal_tree
from spice.libs.ilp.multi.solve_model import get_multiple_trees
from spice.libs.ilp.utils import (check_for_duplicates, check_wgd_presence, concat_dfs, convert_long_to_wide,
                                  process_medicc_initialise)
from spice.libs.postprocess.fill_gaps import add_focal_events_to_eventsdf, fill_removed_segments, split_ilp_events
from spice.libs.postprocess.solution_probabilities import get_solution_probabilities


def save_table(table: Optional[pd.DataFrame], file_name: str, output_dir: Optional[str], index: bool = False) -> None:
    """Save `table` as a tab-separated file in `output_dir`, unless either is None (output is not saved)."""
    if (table is not None) and (output_dir is not None):
        table.to_csv(os.path.join(output_dir, file_name), sep='\t', index=index)


def run_spice_bootstrap(leaves_df: pd.DataFrame,
                        min_seg_size: float = 5e4,
                        max_n_segs: Optional[int] = None,
                        keep_consec_segs: bool = False,
                        allele_specific: bool = True,
                        phase_alleles: bool = False,
                        leaf_dependent_phasing: bool = True,
                        max_cn: int = 8,
                        focal_max_cn: Optional[int] = None,
                        verbose: bool = False,
                        rs: int = 1,
                        n_threads: int = 1,
                        time_limit: float = 5,
                        weight_wgd: bool = True,
                        wgd_fract_genome: float = 0.5,
                        multiple_solutions: bool = True,
                        solution_pool_size: int = 1000,
                        max_cutoff: float = 2,
                        n_bootstraps: int = 1000,
                        fixed_topology: Union[bool, tuple] = False,
                        topology_prop_limit: float = 0.025,
                        topology_topn: int = 10,
                        initnodes: Optional[pd.DataFrame] = None,
                        debug: bool = False,
                        rm_small_segs: bool = False,
                        rank_multiple_trees: bool = True,
                        output_type: str = 'save',
                        output_dir: Optional[str] = None
                        ) -> Optional[dict[str, pd.DataFrame]]:
    '''Main function to run SPICE with bootstrapping of input data and ILP topologies.

    See the module docstring for the steps. Unless `fixed_topology` is given, the topologies used in the ILP are the
    `topology_topn` most frequent bootstrapped topologies, plus any found in at least `topology_prop_limit` of the
    replicates. Each topology gets a share of `solution_pool_size` in proportion to its frequency among them (at least
    1), and a share of `time_limit`: half is split equally between the topologies and half in proportion to their
    frequency. Each topology's share is then split 30% / 70% between the optimal and the multiple-solutions ILP. If the
    enumeration for a topology does not complete to optimality and finds fewer solutions than its share, the same
    proportion of solutions is kept for every topology (`concat_dfs`).

    Parameters
    ----------
    leaves_df : pd.DataFrame
        Leaf copy number profiles, one row per node, chromosome, allele and bin, with columns 'Node', 'Chr', 'Start',
        'End', 'Allele' ('A' or 'B', with both alleles of each bin) and 'CN'. In total copy number mode, 'Allele' may
        be left out; if given, alleles A and B are summed. Every leaf must have the same bins. Other columns are
        ignored with a warning. If `fixed_topology` is False, leaves named by an integer or a string of digits (e.g.
        4 or '4') are renamed 'clone_<node>' (the caller's DataFrame is not modified).
    min_seg_size : float, optional
        Minimum segment size to consider in the ILP (interval model), any segments smaller than this are reconstructed
        using the alternative focal evolutionary model. Ignored when `max_n_segs` is given (default: 5e4).
    max_n_segs : int, optional
        Maximum number of segments to consider in the ILP (interval model), the largest N segments are inferred using
        the ILP model and the remaining segments are reconstructed using the alternative focal evolutionary model
        (default: None, i.e. no limit).
    keep_consec_segs : bool, optional
        If True, some small segments are moved into the ILP so that every segment left to the focal evolutionary
        model has a neighbouring segment in the ILP. With max_n_segs, only the fewest needed are chosen: every third
        segment of each run of consecutive small segments. With min_seg_size, every small segment next to another
        small segment is moved, so only isolated small segments are left to the focal evolutionary model. This
        increases the number of segments in the ILP model (default: False).
    allele_specific : bool, optional
        If True, each allele's copy number is modelled separately; if False, total copy number is modelled throughout
        the tree (default: True).
    phase_alleles : bool, optional
        Whether to phase alleles prior to running SPICE, this will attempt to phase alleles based on the input copy
        number profiles using the phasing algorithm from CHISEL (Zaccaria & Raphael, Nat Biotechnol. 2021) (default:
        False).
    leaf_dependent_phasing : bool, optional
        Whether to prevent the independent phasing of copy number segments in different input leaf profiles (default:
        True, i.e. phasing of a copy number segment occurs in all leaf nodes).
    max_cn : int, optional
        Maximum copy number to consider in the ILP model (default: 8).
    focal_max_cn : int, optional
        Maximum copy number to consider for the focal evolutionary model (default: None, i.e. same max_cn as ILP model).
    verbose : bool, optional
        If True, log at TRACE level and print the Gurobi log and Pyomo's solve timings (default: False).
    rs : int, optional
        Random seed to use for bootstrapping and ILP solver (default: 1).
    n_threads : int, optional
        Number of threads to use for the ILP solver, and number of worker processes for the bootstrapping and the
        focal evolutionary model (default: 1).
    time_limit : float, optional
        Time limit in hours for the ILP solves, shared between the topologies (see above) (default: 5).
    weight_wgd : bool, optional
        Whether to apply a weight to WGD events (equivalent to the highest possible CNA amplitude, i.e. the max_cn
        value) in the ILP objective function and the bootstrap distances (default: True).
    wgd_fract_genome : float, optional
        The minimum fraction of each allele (if allele_specific==True), or the genome, that must be affected by WGD
        events for a WGD called by the optimal ILP to be kept (default: 0.5).
    multiple_solutions : bool, optional
        Whether to enumerate multiple solutions. If False, only the optimal ILP is run, and only the topology with the
        lowest optimal cost is returned (default: True).
    solution_pool_size : int, optional
        The number of solutions to enumerate if multiple_solutions is True, shared between the topologies (default:
        1000).
    max_cutoff : float, optional
        The maximum cost cutoff for enumerating solutions in the multiple solutions ILP, as a multiple of the optimal
        cost (default: 2, i.e. only solutions with cost <= 2x the optimal cost will be enumerated).
    n_bootstraps : int, optional
        The number of bootstraps to run when inferring the phylogenetic tree topology. This parameter is ignored if a
        fixed topology is provided (default: 1000).
    fixed_topology : bool or tuple of tuples, optional
        A fixed topology for the inferred phylogenetic tree, represented as a tuple of tuples (Parent, Child)
        representing the tree edges. If provided, no inference of topologies will be performed and only this topology
        will be used in the ILP model (default: False, i.e. topologies are inferred).
    topology_prop_limit : float, optional
        The proportion of bootstrapped trees that must have a given topology for that topology to be included in the
        ILP model, if it is not among the `topology_topn` most frequent (default: 0.025, i.e. topologies present in at
        least 2.5% of bootstrapped trees are included).
    topology_topn : int, optional
        The number of most frequent bootstrapped topologies that are always included in the ILP model (default: 10).
    initnodes : pd.DataFrame, optional
        DataFrame containing initial node copy number profiles to use for warm-starting the ILP solver (i.e. from
        MEDICC2), with columns 'Node', 'Chr', 'Start', 'End', 'Allele', 'CN'. Note: segments must be defined the same
        in the leaves_df and initnodes Dataframes. 'Allele' is handled as for `leaves_df`: in total copy number mode
        it may be left out, and if given, alleles A and B are summed. If initnodes is provided, fixed_topology must be
        provided with the corresponding topology (default: None, i.e. no initial node profiles provided).
    debug : bool, optional
        Whether SPICE is being run in debug mode, which outputs all variables from the ILP model, the topologies
        used and the position mapping (default: False).
    rm_small_segs : bool, optional
        Whether to filter out small segments from the data (where small segments are defined using the min_seg_size
        or max_n_segs parameters), otherwise the evolution of these segments is inferred using the focal evolutionary
        model (default: False).
    rank_multiple_trees : bool, optional
        Whether to calculate phylogeny probabilities and rank alternative phylogenies. Ignored, with a warning, if
        multiple_solutions is False (default: True).
    output_type : {'save', 'return', 'both'}, optional
        Whether to save the output to output_dir, return it, or both (default: 'save').
    output_dir : str, optional
        The directory in which to save the output of SPICE, the log file and the Gurobi node files; created if it
        does not exist. Only used if output_type != 'return' (default: None, i.e. the current directory).

    Returns
    -------
    dict or None
        Only returned when output_type != 'save': a dictionary of SPICE DataFrames, leaving out any that are None.
        Solutions are named 'o<n>' (the optimal solution of topology n) and 'm<n>' (the multiple solutions, numbered
        across topologies), and 'tree_id' is the topology. Internal nodes are named '0', '1', ... ('0' is the fixed
        diploid root); leaves keep their input names, except that a name of digits (e.g. 4 or '4') becomes 'clone_4'
        if the topology was inferred. When saving, each table is written to output_dir under the file name in
        brackets, plus 'spice.log'. Keys:

        - total_nodes_optimal ['spice_total_nodes_optimal.tsv.gz'] : one row per optimal solution, node and merged
          segment (consecutive input bins with the same copy numbers in every leaf, from 'Start' to 'End'), with each
          allele's copy number ('cnA' / 'cnB', or 'CN' in total copy number mode) and the gain and loss amplitudes on
          the branch into the node ('GainsA' / 'GainsB', 'DeletionsA' / 'DeletionsB'). The optimal solutions are used
          to warm start the enumeration of alternative solutions. If small segments were filled, 'cn_model' is
          'ilp_model' (ILP) or 'focal_model' (focal evolutionary model).
        - total_nodes_multisol ['spice_total_nodes.tsv.gz'] : the same for the multiple solutions.
        - total_edges_optimal ['spice_total_edges_optimal.tsv.gz'] : one row per optimal solution and edge, with the
          number of events ('Events' = 'Gains' + 'Deletions') and WGD on the edge, whether it is the trunk
          ('is_trunk': the edge from the diploid root to the MRCA), and the solution's costs, repeated on each of its
          edges: 'Total_cost' (number of events in the ILP), 'Gurobi_cost' (ILP objective, including WGDs and, for
          optimal solutions, the tie-breaking term) and, if small segments were filled, 'Total_cost_focal' /
          'Total_cost_focal_interval' (including the filled segments).
        - total_edges_multisol ['spice_total_edges.tsv.gz'] : the same for the multiple solutions.
        - total_events ['spice_total_events.tsv.gz'] : one row per copy number event in the multiple solutions: the
          node at the end of its branch, the allele ('A' / 'B'; allele-specific mode only), the event type ('Event'),
          the coordinates it spans ('Start' / 'End') and its amplitude ('nEvents').
            - ILP events are split at the small segments left out of the ILP (filled, or removed with
              `rm_small_segs`), so that each row only covers segments inferred by the ILP: an ILP event over five
              bins with the third left out is two rows, over the first two and the last two bins.
            - 'event_id' identifies each event within a solution, as
              '<model>_<node>.<chr>.<allele>.<event>.<Start>_<End>' with the coordinates before splitting (e.g.
              'ilp_1.1.A.Gain.1000000_5000000'; in total copy number mode without the allele, e.g.
              'ilp_1.1.Gain.1000000_5000000'); the rows of a split event share one ID. 'nEvents' summed over the
              unique 'event_id's of a solution gives its 'Total_cost' or, if small segments were filled, its
              'Total_cost_focal'. See `split_ilp_events` for how to parse the ID.
            - If small segments were filled, the table also holds the events of the focal evolutionary model, and
              'cn_model' is 'ilp_model' for an interval event from the ILP, or 'focal_model' for an event at one
              filled segment (its full change from the parent; 'Gain' / 'Deletion' also on WGD branches).
        - solution_probabilities ['spice_solutions_ranked.tsv.gz'] : one row per multiple solution, with its geometric
          weight ('geometric_weight', from 'Total_cost', so leaving out the events of any filled small segments), its
          topology, copy number (over the merged segments) and solution log probabilities, its topology's bootstrap
          frequency ('bootstrap_frequency': the fraction of the `n_bootstraps` replicates that gave it, NaN for a fixed
          topology), its rank ('solution_rank', by 'solution_log_prob', the sum of the topology and copy number log
          probabilities) and its topology's rank ('topology_rank', by 'topology_log_prob'); 1 is the most probable.
        - cn_probabilities ['spice_cn_probabilities.tsv.gz'] : for each topology ('tree_id'), one row per internal
          node, merged segment ('Start' / 'End'), allele ('A' / 'B') and copy number, with the probability of that
          copy number given that topology. 'top_ranked_topology' is True for the most probable topology
          ('topology_rank' 1) and 'top_ranked_solution' for the topology of the top-ranked solution
          ('solution_rank' 1); if several topologies are tied, all of them.
        - total_debug ['spice_debug.tsv.gz'] : if debug is True, one row per ILP variable and one column per solution.
        - topologies ['spice_topologies.tsv.gz'] : if debug is True, one row per topology used and edge ('Parent' /
          'Child'), with its bootstrap frequency, its share of the solutions and its time limits. Its 'topology_id' is
          equivalent to 'tree_id' in the other tables.
        - solve_info ['spice_solve_info.tsv'] : 'parameter' / 'value' pairs: solve time, model status, gap and cost of
          each topology's ILPs ('tree_id'), plus the total run time and random seed.
        - mapping ['spice_mapping.tsv.gz'] : if debug is True, one row per input bin and allele ('A' / 'B'), with its
          'Start' / 'End', linking SPICE's internal positions: 'origPos' (the input bin, numbered from 1 on each
          chromosome), 'mergePos' (the merged segment, as in the node tables), 'filteredPos' (NaN for segments
          inferred using the focal evolutionary model) and 'finalPos' (the ILP segment).
    '''

    assert output_type in ('save', 'return', 'both'), (f"output_type must be 'save', 'return' or 'both', "
                                                       f"not {output_type!r}")
    save_output = output_type != 'return'
    if save_output:
        if output_dir is None:
            output_dir = './'
        elif not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = None

    logger, _ = initialize_logger(verbose, output_dir)

    # starting memory monitoring
    tracemalloc.start()

    # starting time monitoring
    run_start_time = datetime.datetime.now()
    time_limit = time_limit * 60 * 60

    # PROCESS INPUT DATA
    logger.step('Processing input data')
    if not fixed_topology:
        # Leaves named by an integer (4 or '4') could clash with the internal node IDs, so they become 'clone_<node>'
        node_names = leaves_df['Node'].astype(str)
        leaves_df = leaves_df.assign(Node=np.where(node_names.str.fullmatch(r'\d+'), 'clone_' + node_names,
                                                   leaves_df['Node']))
    (bootstrap_leaves, leaves,
     node_name_mapping, all_leaves, pos_mapping) = process_spice_input(leaves_df, max_cn, phase_alleles,
                                                                       allele_specific, leaf_dependent_phasing,
                                                                       min_seg_size, max_n_segs, keep_consec_segs,
                                                                       focal_max_cn, fixed_topology, logger)
    pos_seg_sizes = (pos_mapping
                     .assign(segsize=pos_mapping.End - pos_mapping.Start)
                     .groupby(['Allele', 'Chr', 'finalPos'])['segsize']
                     .sum()
                     .rename_axis(('Allele', 'Chr', 'Pos')))
    logger.info('Input data valid')

    # The position mapping is only output in debug mode: 'origPos', 'mergePos', ... are SPICE's internal positions
    output_mapping = sort_by_columns(pos_mapping, ['Chr', 'Allele', 'origPos']) if debug else None
    save_table(output_mapping, 'spice_mapping.tsv.gz', output_dir)

    # RUN BOOTSTRAPPING
    if not fixed_topology:
        logger.step('Bootstrapping topologies')
        res = tree_bootstrapping(n_bootstraps, bootstrap_leaves, allele_specific, weight_wgd, max_cn, rs,
                                 njobs=n_threads)
        use_topologies = ((res[['tree', 'topology_id']].value_counts() / n_bootstraps)
                          .reset_index()
                          .rename(columns={'count': 'proportion'})
                          .sort_values(by='proportion', ascending=False))
        top_n_topologies = use_topologies.head(topology_topn)['topology_id'].values
        use_topologies = (use_topologies[(use_topologies['proportion'] >= topology_prop_limit)
                                         | (use_topologies['topology_id'].isin(top_n_topologies))]
                          ['topology_id'].unique())

        if res.topology_id.nunique() - len(use_topologies) > 0:
            logger.warning('Only the top {} topologies, plus those with >={}% frequency, will be used in the ILP : '
                           'removing {} topologies'.format(topology_topn, int(topology_prop_limit * 100),
                                                           len(res['topology_id'].unique()) - len(use_topologies)))

        res_use = res[res['topology_id'].isin(use_topologies)]
        # Stable, so that tied topologies keep their index order: the loop below fills the per-topology lists in row
        # order, and the single-solution mode picks from them by index label
        unique_topologies = ((res_use[['tree', 'topology_id']].value_counts() / len(res_use))
                             .reset_index()
                             .rename(columns={'count': 'proportion'})
                             .sort_values(by='proportion', ascending=False, kind='stable'))
        unique_topologies['n_solutions'] = np.maximum(
            np.round(unique_topologies['proportion'] * solution_pool_size).astype(int), 1)
        unique_topologies['tree_tl'] = ((time_limit / (len(unique_topologies) * 2))
                                        + (unique_topologies['proportion'] * (time_limit / 2)))
        unique_topologies['optimal_tl'] = (unique_topologies['tree_tl'] * 0.3).astype(int)
        unique_topologies['multiple_tl'] = (unique_topologies['tree_tl'] * 0.7).astype(int)
        unique_topologies['bootstrap_frequency'] = (unique_topologies['topology_id']
                                                    .map(res['topology_id'].value_counts() / n_bootstraps))
        logger.info('Bootstrapping complete : {} unique topologies'.format(len(unique_topologies)))
    else:
        assert isinstance(fixed_topology, tuple), ('Fixed topology must be a tuple of tuples representing the tree '
                                                   'edges, e.g. (Parent, Child)')
        fixed_topology = [(node_name_mapping[i], node_name_mapping[j]) for i, j in fixed_topology]
        unique_topologies = pd.DataFrame({'tree': str(fixed_topology), 'topology_id': 1, 'proportion': 1.0,
                                          'n_solutions': solution_pool_size, 'tree_tl': time_limit,
                                          'optimal_tl': int(time_limit * 0.3), 'multiple_tl': int(time_limit * 0.7),
                                          'bootstrap_frequency': np.nan},
                                         index=[0])
        unique_topologies['tree'] = unique_topologies['tree'].apply(lambda x: eval(x))

    if debug:
        mapping = {v: k for k, v in node_name_mapping.items()}
        topologies = unique_topologies.copy().explode('tree')
        topologies['Parent'], topologies['Child'] = zip(*topologies['tree'])
        topologies['Parent'] = topologies['Parent'].map(mapping, na_action='ignore').astype(str)
        topologies['Child'] = topologies['Child'].map(mapping, na_action='ignore').astype(str)
        save_table(topologies, 'spice_topologies.tsv.gz', output_dir)
    else:
        topologies = None

    logger.step(f'Running SPICE ILP: total time limit = {time_limit}s')

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
        assert fixed_topology, 'If initialising internal copy number profiles, the topology must be fixed.'
        initnodes = standardise_alleles(initnodes, allele_specific, 'initnodes', logger)
        opt_init_dict = process_medicc_initialise(initnodes, pos_mapping, node_name_mapping, allele_specific)
        logger.info('CN states provided to initialise optimal ILP, running with warmstart.')
    else:
        opt_init_dict = None

    # RUN OPTIMAL ILP
    for i, row in unique_topologies.iterrows():
        logger.info(f'Running SPICE ILP: topology {i + 1} of {len(unique_topologies)} with {row.n_solutions} '
                    f'solutions : time limit = {row.tree_tl}s')
        (optimal_nodes, optimal_edges, optimal_solve_info,
         optimal_model, optimal_debug) = get_optimal_tree(leaves, max_cn=max_cn, verbose=verbose,
                                                          allele_specific=allele_specific, tree_edges=row.tree,
                                                          time_limit=row.optimal_tl, weight_wgd=weight_wgd,
                                                          init_model=opt_init_dict, nthreads=n_threads, mem=0.1,
                                                          debug=debug, rs=rs, logger=logger, output_dir=output_dir)

        optimal_edges, optimal_model = check_wgd_presence(optimal_nodes, optimal_edges, optimal_model, pos_seg_sizes,
                                                          node_name_mapping, wgd_fract_genome, logger)
        optimal_run_status = (optimal_solve_info[optimal_solve_info.parameter == 'optimal_tree_model_status']
                              .value
                              .values[0])
        optimal_cost = optimal_solve_info[optimal_solve_info.parameter == 'optimal_tree_total_cost'].value.values[0]
        if optimal_run_status == 'optimal':
            logger.info(f'SPICE ILP completed to optimality for topology {i + 1} of {len(unique_topologies)}')
        else:
            logger.warning(f'SPICE ILP did not complete to optimality for topology {i + 1} of '
                           f'{len(unique_topologies)}: status = {optimal_run_status}')
            optimal_gap = optimal_solve_info[optimal_solve_info.parameter == 'optimal_tree_gap'].value.values[0]
            if optimal_gap > 0:
                logger.warning(f'Optimal solution not reached for topology {i + 1} of {len(unique_topologies)}: '
                               f'Gap of {optimal_gap} found - consider increasing time limit')

        optimal_nodes['Solution'] = f'o{i + 1}'
        optimal_nodes['tree_id'] = i + 1
        total_opt_nodes.append(optimal_nodes)

        optimal_edges['Solution'] = f'o{i + 1}'
        optimal_edges['tree_id'] = i + 1
        total_opt_edges.append(optimal_edges)

        optimal_solve_info['tree_id'] = i + 1
        total_opt_solve_info.append(optimal_solve_info)

        if optimal_debug is not None:
            optimal_debug.columns = [f'o{i + 1}']
            total_opt_debug.append(optimal_debug)

        unique_topologies.loc[i, 'n_events'] = optimal_cost

        # RUN ENUMERATE SOLUTIONS ILP
        if multiple_solutions:
            (multisol_nodes, multisol_events, multisol_edges,
             multisol_solve_info, multisol_debug) = get_multiple_trees(leaves, max_cn=max_cn, verbose=verbose,
                                                                       allele_specific=allele_specific,
                                                                       tree_edges=row.tree, time_limit=row.multiple_tl,
                                                                       weight_wgd=weight_wgd,
                                                                       solution_pool_size=row.n_solutions,
                                                                       init_model=optimal_model,
                                                                       sol_cutoff=optimal_cost * max_cutoff,
                                                                       nthreads=n_threads, mem=0.1, debug=debug, rs=rs,
                                                                       logger=logger, output_dir=output_dir)

            multisol_run_status = (multisol_solve_info[multisol_solve_info.parameter == 'multi_solution_model_status']
                                   .value
                                   .values[0])
            n_solutions = multisol_edges.Solution.nunique()
            if multisol_run_status == 'optimal':
                logger.info(f'Enumeration of solutions completed to optimality for topology {i + 1} of '
                            f'{len(unique_topologies)}')
            else:
                multisol_gap = (multisol_solve_info[multisol_solve_info.parameter == 'multi_solution_gap']
                                .value
                                .values[0])
                logger.warning(f'Enumeration of solutions did not complete to optimality for topology {i + 1} of '
                               f'{len(unique_topologies)}: Model status = {multisol_run_status} - gap of '
                               f'{multisol_gap} found - consider increasing time limit')
                if n_solutions < row.n_solutions:
                    logger.warning(f'Only {n_solutions} of expected {row.n_solutions} solutions found for topology '
                                   f'{i + 1} of {len(unique_topologies)} - consider increasing time limit or reducing '
                                   f'solution pool size')
                    missing_sols = 1 - (n_solutions / row.n_solutions)
                    prop_missing_sols = np.max([missing_sols, prop_missing_sols])
            if ((n_solutions / row.n_solutions) < 1) and (1 - (n_solutions / row.n_solutions) == prop_missing_sols):
                logger.warning(f'{prop_missing_sols * 100}% of solutions from previous topologies were not enumerated '
                               f'within the time limit, the number of solutions from other topologies will be '
                               f'adjusted to match. The total number of solutions will be reduced from '
                               f'{solution_pool_size} to {round(solution_pool_size * (1 - prop_missing_sols))}')

            multisol_nodes['tree_id'] = i + 1
            total_multisol_nodes.append(multisol_nodes)

            multisol_events['tree_id'] = i + 1
            total_multisol_events.append(multisol_events)

            multisol_edges['tree_id'] = i + 1
            total_multisol_edges.append(multisol_edges)

            multisol_solve_info['tree_id'] = i + 1
            total_multisol_solve_info.append(multisol_solve_info)

            if multisol_debug is not None:
                multisol_debug['tree_id'] = i + 1
                total_multisol_debug.append(multisol_debug)

    # PROCESS ALL OUTPUT
    if multiple_solutions:
        total_multisol_nodes = concat_dfs(total_multisol_nodes, 1 - prop_missing_sols)
        total_nodes = pd.concat(total_opt_nodes + [total_multisol_nodes]).reset_index(drop=True)
        total_multisol_edges = concat_dfs(total_multisol_edges, 1 - prop_missing_sols)
        total_edges = pd.concat(total_opt_edges + [total_multisol_edges]).reset_index(drop=True)
        total_events = concat_dfs(total_multisol_events, 1 - prop_missing_sols)
        total_solve_info = pd.concat(total_opt_solve_info + total_multisol_solve_info).reset_index(drop=True)
        if len(total_multisol_debug) > 0:
            total_multisol_debug = (concat_dfs(total_multisol_debug, 1 - prop_missing_sols)
                                    .pivot_table(index='index', columns='Solution', values='Value'))
            # Order the solutions by number ('m2' before 'm10'), not as text
            total_multisol_debug = total_multisol_debug[sorted(total_multisol_debug.columns, key=lambda x: int(x[1:]))]
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
    save_table(total_debug, 'spice_debug.tsv.gz', output_dir, index=True)

    solution_topology_mapping = total_edges[['Solution', 'tree_id']].drop_duplicates()
    solution_topology_mapping = dict(zip(solution_topology_mapping.Solution, solution_topology_mapping.tree_id))

    ilp_end_time = datetime.datetime.now()
    total_ilp_time = (ilp_end_time - ilp_start_time).total_seconds()
    logger.info(f'SPICE ILP complete: total ILP time = {np.round(total_ilp_time, 2)}')

    # MAP NODES AND SOLUTIONS BACK TO ORIGINAL (FROM ILP ENCODING)
    mapping = {v: k for k, v in node_name_mapping.items()}
    total_nodes['Node'] = total_nodes['Node'].map(mapping, na_action='ignore').astype(str)
    total_nodes['tree_id'] = total_nodes['Solution'].map(solution_topology_mapping)
    total_edges['Parent'] = total_edges['Parent'].map(mapping, na_action='ignore').astype(str)
    total_edges['Child'] = total_edges['Child'].map(mapping, na_action='ignore').astype(str)
    total_edges['tree_id'] = total_edges['Solution'].map(solution_topology_mapping)
    if total_events is not None:
        total_events['Node'] = total_events['Node'].map(mapping, na_action='ignore').astype(str)
        total_events['tree_id'] = total_events['Solution'].map(solution_topology_mapping)

    # RUN SMALL SEGMENTS INFERENCE
    if (len(all_leaves[all_leaves.filter_seg]) > 0) and (not rm_small_segs):
        logger.step('Filling CN small segments using the focal evolutionary model')
        all_leaves['Node'] = all_leaves['Node'].astype(str)
        total_nodes, total_edges, focal_events = fill_removed_segments(total_nodes, total_edges, all_leaves,
                                                                       pos_mapping, max_cn, n_threads,
                                                                       focal_maxcn=focal_max_cn, logger=logger,
                                                                       allele_specific=allele_specific)
        logger.info('Filling CN small segments complete')
    else:
        focal_events = None
        pos_mapping['Allele'] = pos_mapping['Allele'].map({'A': 0, 'B': 1})
        total_nodes = convert_long_to_wide(expand_segments(total_nodes, pos_mapping, 'finalPos', 'mergePos'))
        total_edges = total_edges[['tree_id'] + [i for i in total_edges.columns if i != 'tree_id']]

    # CONVERT SPICE RUN GENOME POSITIONS BACK TO ORIGINAL POSITIONS
    col_order = ['tree_id', 'Solution', 'Node', 'Chr', 'Start', 'End', 'startPos', 'endPos', 'Allele', 'Event',
                 'nEvents']
    if total_events is not None:
        total_events = map_to_original_positions(total_events, pos_mapping, 'finalPos')[col_order]
        if focal_events is not None:
            total_events = add_focal_events_to_eventsdf(total_events, focal_events, pos_mapping,
                                                        allele_specific=allele_specific)
            cost_col = 'Total_cost_focal'
        else:
            total_events = split_ilp_events(total_events, pos_mapping, allele_specific=allele_specific)
            cost_col = 'Total_cost'
        event_totals = (total_events
                        .drop_duplicates(['Solution', 'event_id'])
                        .groupby('Solution')['nEvents']
                        .sum())
        edge_costs = (total_edges[total_edges.Solution.str.contains('m')]
                      .groupby('Solution')[cost_col]
                      .first())
        mismatch = edge_costs.index[~np.isclose(event_totals.reindex(edge_costs.index, fill_value=0), edge_costs)]
        assert len(mismatch) == 0, (f'Events in total_events do not sum to {cost_col!r} for {len(mismatch)} '
                                    f'solutions, e.g. {list(mismatch[:3])}')

    # MARK THE TRUNK (THE EDGE FROM THE DIPLOID ROOT TO THE MRCA)
    total_edges['is_trunk'] = ~total_edges['Parent'].isin(total_edges['Child'])

    # ENSURE THAT EVERY SOLUTION IS UNIQUE
    if multiple_solutions:
        total_nodes, total_edges, total_events = check_for_duplicates(total_nodes, total_edges, total_events, logger)

    # SPLIT INTO OPTIMAL AND MULTIPLE SOLUTIONS
    total_nodes_optimal = total_nodes[total_nodes.Solution.str.contains('o')]
    total_nodes_multisol = total_nodes[total_nodes.Solution.str.contains('m')].copy()
    total_edges_optimal = total_edges[total_edges.Solution.str.contains('o')]
    total_edges_multisol = total_edges[total_edges.Solution.str.contains('m')].copy()

    # LABEL ALLELES 'A' / 'B' (0 / 1 IS ONLY THE ILP ENCODING), NAME THE TOTAL COPY NUMBER 'CN', SORT AND SAVE THE
    # OUTPUT TABLES (the node tables stay at the merged segments, given by 'Start' / 'End'). The multiple-solutions node
    # table keeps the internal 'Pos', which the ranking below uses, until it is returned; it is saved without it.
    total_nodes_optimal = sort_by_columns(total_nodes_optimal.drop(columns='Pos').rename(columns={'cn': 'CN'}),
                                          ['tree_id', 'Solution', 'Node', 'Chr', 'Start'])
    total_nodes_multisol = sort_by_columns(total_nodes_multisol.rename(columns={'cn': 'CN'}),
                                           ['tree_id', 'Solution', 'Node', 'Chr', 'Start'])
    total_edges_optimal = sort_by_columns(total_edges_optimal, ['tree_id', 'Solution', 'Parent', 'Child'])
    total_edges_multisol = sort_by_columns(total_edges_multisol, ['tree_id', 'Solution', 'Parent', 'Child'])
    if total_events is not None:
        if allele_specific:
            total_events['Allele'] = total_events['Allele'].map({0: 'A', 1: 'B'})
            total_events = sort_by_columns(total_events.drop(columns=['startPos', 'endPos']),
                                           ['Solution', 'Node', 'Chr', 'Allele', 'Start', 'End', 'Event'])
        else:
            total_events = sort_by_columns(total_events.drop(columns=['startPos', 'endPos', 'Allele']),
                                           ['Solution', 'Node', 'Chr', 'Start', 'End', 'Event'])
    save_table(total_nodes_optimal, 'spice_total_nodes_optimal.tsv.gz', output_dir)
    save_table(total_nodes_multisol.drop(columns='Pos'), 'spice_total_nodes.tsv.gz', output_dir)
    save_table(total_edges_optimal, 'spice_total_edges_optimal.tsv.gz', output_dir)
    save_table(total_edges_multisol, 'spice_total_edges.tsv.gz', output_dir)
    save_table(total_events, 'spice_total_events.tsv.gz', output_dir)

    # RANK MULTIPLE SOLUTIONS (on the merged segments, so that splitting a segment into input bins doesn't change it)
    solution_probabilities = None
    cn_probabilities = None
    if rank_multiple_trees:
        if not multiple_solutions:
            logger.warning('Multiple solutions parameter set to False, no multiple trees to rank')
        else:
            logger.step('Calculating solution probabilities and ranking multiple solutions')
            topology_frequency = dict(zip(unique_topologies.index + 1, unique_topologies['bootstrap_frequency']))
            (solution_probabilities,
             cn_probabilities) = get_solution_probabilities(total_edges_multisol, total_nodes_multisol,
                                                            topology_frequency, logger,
                                                            allele_specific=allele_specific)
            logger.info('Ranking multiple solutions complete')
    total_nodes_multisol = total_nodes_multisol.drop(columns='Pos')  # 'Pos' is internal (saved without it above)

    # LABEL ALLELES 'A' / 'B' AND SORT THE RANKING TABLES (cn_probabilities stays at the merged segments)
    if solution_probabilities is not None:
        solution_probabilities = sort_by_columns(solution_probabilities, ['solution_rank', 'tree_id', 'Solution'])
        if 'Allele' in cn_probabilities.columns:
            cn_probabilities['Allele'] = cn_probabilities['Allele'].map({'cnA': 'A', 'cnB': 'B'})
        cn_probabilities = sort_by_columns(cn_probabilities.drop(columns='Pos'),
                                           ['tree_id', 'Node', 'Chr', 'Start', 'Allele', 'CN'])
    save_table(solution_probabilities, 'spice_solutions_ranked.tsv.gz', output_dir)
    save_table(cn_probabilities, 'spice_cn_probabilities.tsv.gz', output_dir)

    # CALCULATE RUN INFORMATION
    run_time = (datetime.datetime.now() - run_start_time).total_seconds()
    logger.info(f'Total SPICE run time: {np.round(run_time, 2)}s')

    # CALCULATE MEMORY INFORMATION
    peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # Convert to MB
    tracemalloc.stop()
    logger.info(f'Peak memory usage: {np.round(peak_mem, 2)} MB')

    run_info = (pd.DataFrame({'total_run_time': run_time, 'random_seed': rs}, index=['value'])
                .T
                .reset_index()
                .rename(columns={'index': 'parameter'}))
    run_info = pd.concat([total_solve_info, run_info]).reset_index(drop=True)
    save_table(run_info, 'spice_solve_info.tsv', output_dir)
    if save_output:
        for node_dir in glob.glob(os.path.join(output_dir, 'grbnodes*/')):
            shutil.rmtree(node_dir)

    logger.step('SPICE complete')
    close_handlers(logger)

    # RETURN SPICE OUTPUTS
    if output_type != 'save':
        results = {'total_nodes_optimal': total_nodes_optimal, 'total_edges_optimal': total_edges_optimal,
                   'total_edges_multisol': total_edges_multisol, 'total_nodes_multisol': total_nodes_multisol,
                   'total_events': total_events, 'solution_probabilities': solution_probabilities,
                   'cn_probabilities': cn_probabilities,
                   'total_debug': total_debug, 'topologies': topologies, 'solve_info': run_info,
                   'mapping': output_mapping}
        return {k: v for k, v in results.items() if v is not None}
