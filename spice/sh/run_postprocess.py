import pyomo.environ as pe
import numpy as np
import time
import datetime
import sys
import shutil
import argparse
import os
import timeit
import logging
# import colorlog

sys.path.append('/camp/home/bunkuma/working/bunkuma/repos/abi-spice/src/')

from spice.spice_bootstrap import *
from spice.io.convert_input import *
from spice.postprocess.fill_gaps import *
from spice.postprocess.solution_probabilities import *
from spice.io.logging import *

def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t", "True"]:
        return True
    elif value in ["false", "no", "n", "0", "f", 'False']:
        return False

    return False

run_start_time = timeit.default_timer()

parser = argparse.ArgumentParser(description='Performance accuracy evaluation')
# parser.add_argument("--input_file", type=str, default=None, help='Input filepath leaves')
parser.add_argument("--output_dir", type=str, default=None, help='Input dir')
parser.add_argument("--verbose", action='store_true', default=False, help='True, False')
parser.add_argument("--threads", type=int, default=1, help='warm start time limit proportion')
parser.add_argument("--suffix", type=str, default='', help='output_file_suffix')
parser.add_argument("--top_prop", type=float, default=None, help='output_file_suffix')
parser.add_argument("--weight_func", type=str, default='linear_max', help='weight function to use')

args = parser.parse_args()

# leaves = args.input_file
suffix = args.suffix
weight_func = args.weight_func
weight_func = False if weight_func == 'False' else weight_func
out_dir = os.path.join(args.output_dir, 'spice' + suffix)
threads = args.threads
verbose = args.verbose
top_prop = args.top_prop


if out_dir is None:
    out_dir = './'
elif (not os.path.exists(out_dir)):
    os.makedirs(out_dir)
    
logger, logfile = initialize_logger(verbose, out_dir, mode='a')
logger.info('Running SPICE post-processing')
    
logger.info('Processing input data')
total_nodes = pd.read_csv(os.path.join(out_dir, 'spice_total_nodes.tsv.gz'), sep='\t')
total_edges = pd.read_csv(os.path.join(out_dir, 'spice_total_edges.tsv.gz'), sep='\t')
logger.info('Input data valid')

if top_prop is not None:
    logger.info('Filtering trees based on topology proportion value of {}'.format(top_prop))
    tree_props = total_edges[['tree_id', 'Solution']].drop_duplicates().groupby('tree_id')['Solution'].count()
    tree_props = tree_props / tree_props.sum()
    use_trees = list(tree_props[tree_props > top_prop].index)
    total_edges = total_edges[total_edges['tree_id'].isin(use_trees)].copy()
    total_nodes = total_nodes[total_nodes['tree_id'].isin(use_trees)].copy()

# if fill_gaps:
#     leaves_df = pd.read_csv(leaves, sep='\t')
#     spice_input_leaves, all_leaves, final_mapping = process_spice_input(leaves_df, min_seg_size=min_seg_size, max_cn=max_cn)
#     leaves, leaf_name_mapping = df_to_leaves_input(spice_input_leaves, allele_specific=allele_specific)

#     logger.info('Filling CN noise')
#     mapping = {k : leaf_name_mapping[k] if k in leaf_name_mapping.keys() else k for k in total_nodes['Node'].unique()}
#     total_nodes['Node'] = total_nodes['Node'].map(mapping, na_action='ignore')
#     total_edges['Child'] = total_edges['Child'].map(mapping, na_action='ignore')

#     total_nodes = fill_removed_segments(total_nodes, total_edges, all_leaves, final_mapping, max_cn, threads, logger)
#     total_nodes.to_csv(os.path.join(out_dir, 'spice_total_nodes.tsv.gz'), sep='\t', index=False)
#     logger.info('Filling CN noise complete')

# if rank: 
# gap = float(pd.read_csv(os.path.join(out_dir, 'spice_solve_info.tsv'), sep='\t', header=None, index_col=0)[1].to_dict()['gap'])
logger.info('Ranking multiple solutions')
solution_probabilities, chosen_topology_cn_probabilities = get_solution_probabilities(total_edges, total_nodes, logger, weight_func=weight_func, gap=0) 
file_suffix = '' if weight_func == 'linear_max' else '_' + weight_func.replace('_', '') if str(weight_func) != 'False' else '_unweighted'
top_prop_suffix = '' if top_prop is None else '_{}'.format(top_prop)

solution_probabilities.to_csv(os.path.join(out_dir, f'spice_solutions_ranked{file_suffix}{top_prop_suffix}.tsv.gz'), sep='\t', index=False)
chosen_topology_cn_probabilities.to_csv(os.path.join(out_dir, f'spice_cn_probabilities{file_suffix}{top_prop_suffix}.tsv.gz'), sep='\t', index=False)
logger.info('Ranking multiple solutions complete')  

logger.info('SPICE complete')


