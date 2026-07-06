import numpy as np
import sys
import pandas as pd
import argparse
import os
import timeit
import logging

sys.path.append('/camp/home/bunkuma/working/bunkuma/repos/abi-spice/src/')

from spice.spice_bootstrap import *
from spice.io.convert_input import *

def parse_boolean(value):
    value = value.lower()
    if value in ["true", "yes", "y", "1", "t", "True"]:
        return True
    elif value in ["false", "no", "n", "0", "f", 'False']:
        return False
    return False

print('\n======== RUNNING TUMOUR PHYLOGENY RECONSTRUCTION ========\n')

run_start_time = timeit.default_timer()

parser = argparse.ArgumentParser(description='Performance accuracy evaluation')
parser.add_argument("--input_file", type=str, default=None, help='Input filepath leaves')
parser.add_argument("--output_dir", type=str, default=None, help='Input dir')
parser.add_argument("--min_segsize", type=float, default=5e4, help='min_seg_size')
parser.add_argument("--keep_consec_segs", action='store_true', default=False, help='True, False')
parser.add_argument("--max_n_segs", type=int, default=None, help='maximum number of segments')
parser.add_argument("--allele_specific", action='store_true', default=False, help='True, False')
parser.add_argument("--allele_phasing", action='store_true', default=False, help='True, False')
parser.add_argument("--leaf_dependent_phasing", action='store_true', default=True, help='True, False')
parser.add_argument("--max_cn", type=int, default=8, help='maximum CN')
parser.add_argument("--focal_max_cn", type=int, default=None, help='maximum CN for small segs model')
parser.add_argument("--verbose", action='store_true', default=False, help='True, False')
parser.add_argument("--weighted_wgd", action='store_true', default=False, help='True, False')
parser.add_argument("--wgd_fract_genome", type=float, default=0.5, help='Fraction of genome with increased CN to call WGD')
parser.add_argument("--multiple_solutions", action='store_true', default=False, help='True, False')
parser.add_argument("--rank_multiple_trees", action='store_true', default=False, help='True, False')
parser.add_argument("--rm_small_segs", action='store_true', default=False, help='True, False')
parser.add_argument("--solution_pool_size", type=int, default=1000, help='size of solution pool')
parser.add_argument("--debug", action='store_true', default=False, help='options : True, False')
parser.add_argument("--time_limit", type=float, default=5.0, help='ilp time limit')
parser.add_argument("--threads", type=int, default=1, help='warm start time limit proportion')
parser.add_argument("--n_bootstraps", type=int, default=1000, help='number of bootstraps')
parser.add_argument("--fixed_topology", type=str, default='False', help='tuple of edges if topology is fixed otherwise False')
parser.add_argument("--topology_prop_lim", type=float, default=0.025, help='Proportion of top toplogies to consider')
parser.add_argument("--init_nodes_file", type=str, default=None, help='Initialize nodes filepath')
parser.add_argument("--topology_topn", type=int, default=10, help='number of top topologies to consider (alongside topology_prop_lim)')

args = parser.parse_args()

leaves = args.input_file
out_dir = args.output_dir 
min_seg_size = int(args.min_segsize)
max_n_segs = args.max_n_segs if args.max_n_segs is None else int(args.max_n_segs)
keep_consec = args.keep_consec_segs
allele_specific = args.allele_specific 
allele_phasing = args.allele_phasing 
leaf_dependent_phasing = args.leaf_dependent_phasing
max_cn = args.max_cn
focal_max_cn = args.focal_max_cn
verbose = args.verbose
weight_wgd = args.weighted_wgd
wgd_fract_genome = args.wgd_fract_genome
multiple_solutions = args.multiple_solutions
solution_pool_size = args.solution_pool_size
time_limit = args.time_limit
threads = args.threads
rank = args.rank_multiple_trees
debug = args.debug
n_bootstraps = args.n_bootstraps
fixed_topology = args.fixed_topology
top_n = args.topology_topn
top_prop = args.topology_prop_lim
init_nodes = args.init_nodes_file
rm_small_segs = args.rm_small_segs

if out_dir is None:
    out_dir = './'
elif (not os.path.exists(out_dir)):
    os.makedirs(out_dir)

run_details = {'min_seg_size' : min_seg_size, 'max_nsegs' : max_n_segs, 'keep_consec_segs' : keep_consec, 
               'allele_specific' : allele_specific, 'allele_phasing' : allele_phasing, 'n_bootstraps' : n_bootstraps,
               'phasing_type' : np.NaN if allele_phasing == 'fixed' else 'leaf_dependent' if leaf_dependent_phasing else 'leaf_independent', 
               'max_cn' : max_cn, 'focal_max_cn' : focal_max_cn, 'weighted_wgd' : weight_wgd, 'wgd_fract_genome' : wgd_fract_genome,
               'fixed_topology' : fixed_topology, 'topology_top_number' : top_n, 'topology_prop_limit' : top_prop, 
               'nodes_initialised' : False if init_nodes is None else True, 'multiple_solutions' : multiple_solutions, 
               'solution_pool_size' : solution_pool_size, 'rm_small_segs' : rm_small_segs, 'rank_multiple_trees' : rank}

run_detail_df = pd.DataFrame(run_details, index=[0]).T.reset_index()
run_detail_df.to_csv(os.path.join(out_dir, 'run_details.tsv'), header=False, index=False, sep='\t')

print('\n-------- PARAMETERS --------\n')
for k, v in run_details.items():
    print(k + ' : ' + str(v) + '\n')
print('\n\n')

get_sep = (lambda x: '\t' if 'tsv' in x else ',')
leaves = pd.read_csv(leaves, sep=get_sep(leaves))
if init_nodes is not None:
    init_nodes = pd.read_csv(init_nodes, sep=get_sep(init_nodes))

run_spice_bootstrap(leaves, 
                    min_seg_size=min_seg_size,
                    max_n_segs=max_n_segs,
                    keep_consec_segs=keep_consec,
                    allele_specific=allele_specific,
                    phase_alleles=allele_phasing,
                    leaf_dependent_phasing=leaf_dependent_phasing,
                    max_cn=max_cn, 
                    focal_max_cn=focal_max_cn,
                    verbose=verbose,
                    n_threads=threads, 
                    time_limit=time_limit,
                    weight_wgd=weight_wgd,
                    wgd_fract_genome=wgd_fract_genome,
                    multiple_solutions=multiple_solutions,
                    solution_pool_size=solution_pool_size,
                    n_bootstraps=n_bootstraps,
                    fixed_topology=eval(fixed_topology),
                    topology_prop_limit=top_prop,
                    topology_topn=top_n,
                    initnodes=init_nodes,
                    debug=False, 
                    rm_small_segs=rm_small_segs,
                    rank_multiple_trees=rank,
                    output_type='save',
                    output_dir=out_dir)  

print('\n======= COMPLETE =======')

