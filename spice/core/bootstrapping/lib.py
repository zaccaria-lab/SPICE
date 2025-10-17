import numpy as np
import pandas as pd
import time
import itertools
from multiprocessing import Pool
import sys
import datetime

from Bio.Phylo.TreeConstruction import DistanceMatrix as PDistanceMatrix
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from collections import deque

check_cn_same = (lambda profiles, leaves, c, a: np.all([np.equal(profiles[leaves[0]][c][a], profiles[l][c][a]) for l in leaves[1:]]))


def tree_bootstrapping(n_runs, leaves, segsizes, allele_specific, weight_wgd, max_cn, dist_func, rs, njobs=1):
    np.random.seed(rs)
    chrs = list(leaves[list(leaves.keys())[0]].keys())
    use_chrs = [c for c in chrs if ~np.all([check_cn_same(leaves, list(leaves.keys()), c, a) for a in range(1+allele_specific)])] 
    with Pool(processes=njobs, initializer=init_trees, initargs=(leaves, segsizes, use_chrs, max_cn, allele_specific, weight_wgd, dist_func)) as pool:
        jobs = np.random.choice(2**20, n_runs, replace=False)
        bar = ProgressBar(total=len(jobs), length=30, verbose=False)
        progress = (lambda e : bar.progress(advance=True, msg="Pair: {}".format(e)))
        bar.progress(advance=False, msg="Started")
        results = {i+1 : [tree, dm] for i, tree , dm in pool.imap_unordered(get_tree, jobs) if progress(i)}
    results_df = pd.DataFrame([results]).T.reset_index().rename(columns={0 : 'tree', 'index' : 'tree_id'})
    results_df[['tree', 'dm']] = pd.DataFrame(results_df['tree'].tolist(), index=results_df.index)
    unique_trees = {j : i+1 for i, j in enumerate(list(results_df['tree'].value_counts().index))}
    results_df['topology_id'] = results_df['tree'].apply(lambda x: unique_trees[x])
    return results_df.sort_values(by='topology_id')


def get_tree(job):
    np.random.seed(job)
    tree, dm = generate_tree(leaves=LEAVES, segsizes=SEGSIZES, use_chrs=USE_CHRS, allele_specific=ALLELE_SPECIFIC, weight_wgd=WEIGHT_WGD, max_cn=MAX_CN, dist_func=DIST_FUNC)
    return job, tree, dm


def init_trees(_leaves, _segsizes, _use_chrs, _max_cn, _allele_specific, _weight_wgd, _dist_func):
    global LEAVES, SEGSIZES, USE_CHRS, MAX_CN, ALLELE_SPECIFIC, WEIGHT_WGD, DIST_FUNC
    LEAVES = _leaves
    SEGSIZES = _segsizes
    USE_CHRS = _use_chrs
    MAX_CN = _max_cn
    ALLELE_SPECIFIC = _allele_specific
    WEIGHT_WGD = _weight_wgd
    DIST_FUNC = _dist_func
    

def generate_tree(leaves, segsizes, use_chrs, allele_specific, weight_wgd, max_cn, dist_func):
    leaves_new, segsizes_new = generate_bootstrap_leaves(leaves, segsizes, use_chrs) #use_original_leaves(leaves) #! 
    leaves_new, leaf_names_normal = add_fake_normal(leaves_new, allele_specific)
    wgd_cost = max_cn if weight_wgd else 1
    distance_matrix, max_distance = get_distance_matrix(leaves_new, wgd_cost, dist_func, segsizes_new)
    T = make_neighbour_joining_tree(distance_matrix, leaf_names_normal)
    nj_edges, nj_mapping = smallleft_dsf(T)
    return tuple(sorted(nj_edges)), pd.DataFrame(distance_matrix)

def use_original_leaves(leaves): # FOR TESTING DISTANCE METRICS ONLY!
    return {l : {c : [list(np.array(leaves[l][c][a])) for a in range(len(leaves[l][c]))] for c in leaves[l].keys()} for l in leaves.keys()}

def generate_bootstrap_leaves(leaves, segsizes, use_chrs):
    sampled_chrs = np.random.choice(use_chrs, 22, replace=True)
    sample_segs = (lambda n: list(np.sort(np.random.choice(n, n, replace=True))))
    sampled_segs = {i : sample_segs(len(leaves[1][c][0])) for i, c in enumerate(sampled_chrs)} 
    sampled_leaves = {l : {i+1 : [list(np.array(leaves[l][c][a])[sampled_segs[i]]) for a in range(len(leaves[l][c]))] 
                                                                         for i, c in enumerate(sampled_chrs)} 
                                                                         for l in leaves.keys()}
    sampled_segsizes = {i+1 : [list(np.array(segsizes[c][a])[sampled_segs[i]]) for a in range(len(segsizes[c]))] 
                                                                         for i, c in enumerate(sampled_chrs)} 
    return sampled_leaves, sampled_segsizes


def add_fake_normal(leaves, allele_specific):
    n_leaves = len(leaves)
    leaf_names = list(leaves.keys())
    leaf_names_normal = [0] + [i + n_leaves - 1 for i in leaf_names]
    if allele_specific:
        leaves[0] = {chr: [list(np.ones(len(i), dtype=int)) for i in cn] for chr, cn in leaves[1].items()}
    else:
        leaves[0] = {chr: list(np.ones(len(cn), dtype=int)) for chr, cn in leaves[1].items()}
    return leaves, leaf_names_normal

add_wgd_effect = (lambda cell, wgd_add: cell + wgd_add)
add_telomeres = (lambda cell: np.concatenate(([1], np.array(cell), [1])))
calc_bp_l1normdist = (lambda cell_1, cell_2: np.sum(np.abs(np.diff(cell_1) - np.diff(cell_2))))
calc_bp_hammdist = (lambda cell_1, cell_2: np.sum(np.abs(np.diff(cell_1) != np.diff(cell_2))))
calc_hammdist = (lambda cell_1, cell_2, segsizes: np.sum(cell_1 != cell_2))
calc_hammweighteddist = (lambda cell_1, cell_2, segsizes: np.sum((cell_1 != cell_2) * segsizes) / np.sum(segsizes))

def get_breakpoint_distance(cell_1, cell_2, wgd_add=0, dist_func='bpl1norm', segsizes=None):
    cell_1 = np.array(cell_1)
    cell_2 = np.array(cell_2)
    wgd_add_c1 = wgd_add if type(wgd_add) == int else cell_1
    wgd_add_c2 = wgd_add if type(wgd_add) == int else cell_2
    
    dist_funcs = {'bpl1norm' : calc_bp_l1normdist, 'bphamm' : calc_bp_hammdist, 'hamm' : calc_hammdist, 'hammweighted' : calc_hammweighteddist}
    
    if dist_func in ['hamm', 'hammweighted']:
        d1 = dist_funcs[dist_func](cell_2, add_wgd_effect(cell_1, wgd_add_c1), segsizes)
        d2 = dist_funcs[dist_func](cell_1, add_wgd_effect(cell_2, wgd_add_c2), segsizes)
    else:
        d1 = dist_funcs[dist_func](add_telomeres(cell_2), add_telomeres(add_wgd_effect(cell_1, wgd_add_c1)))
        d2 = dist_funcs[dist_func](add_telomeres(cell_1), add_telomeres(add_wgd_effect(cell_2, wgd_add_c2)))
    return np.min([d1, d2]) / 2

def get_WGD_breakpoint_distance(leaves, cell_1, cell_2, wgd_cost, dist_func, segsizes): 
    get_wgd_add_value = (lambda x: x if type(x) == int else 1)
    res = [np.sum(get_breakpoint_distance(leaves[cell_1][ch][a], leaves[cell_2][ch][a], wgd_add, dist_func, segsizes[ch][a]) 
                                        for ch, profile in leaves[cell_1].items()
                                        for a in range(len(profile))) + (get_wgd_add_value(wgd_add) * wgd_cost)
                                    for wgd_add in [0, 1, 2, 3, 'x2']]
    min_res = min(res)
    return min_res


def calculate_distance(leaves, cell_1, cell_2, wgd_cost, dist_func, segsizes): 
    check_profiles_same_len = (lambda leaves, cell_1, cell_2 : np.all([len(leaves[cell_1][ch][allele]) == len(leaves[cell_2][ch][allele]) 
                                                                   for ch, profile in leaves[cell_1].items() 
                                                                   for allele in range(len(profile))]))
    assert (leaves[cell_1].keys() == leaves[cell_2].keys()) & (check_profiles_same_len(leaves, cell_1, cell_2))
    return get_WGD_breakpoint_distance(leaves, cell_1, cell_2, wgd_cost, dist_func, segsizes)


def get_distance_matrix(leaves, wgd_cost, dist_func, segsizes):
    pairs = set((i, j) if i >= j else (j, i) for i, j in itertools.combinations(leaves.keys(), 2))
    distance_matrix = {pair : calculate_distance(leaves, pair[0], pair[1], wgd_cost, dist_func, segsizes) for pair in pairs}
    dm = [[distance_matrix[j, i] for i in sorted(leaves.keys()) if j > i] + [0] for j in sorted(leaves.keys())]
    return dm, max(distance_matrix.values())


def make_neighbour_joining_tree(distance_matrix, leaves):
    T = DistanceTreeConstructor().nj(PDistanceMatrix(names=list(map(str, leaves)), matrix=distance_matrix))
    T.root_with_outgroup(str(0))
    return T


def smallleft_dsf(T):
    nleaves = T.count_terminals()
    mapping, edges = smallleft_dsf_inner(T.root, set(), deque(range(nleaves, 2 * nleaves)))
    nj_edges = [(i, j) for i, j in list(edges) if j != 0]
    nj_mapping = {str(i) : j for i, j in mapping.items() if (str(i) != 'None')}
    return nj_edges, nj_mapping 


def smallleft_dsf_inner(node, internal, leaves):
    if node.is_terminal():
        return {node.name : int(node.name)}, set()
    else:
        assert len(node.clades) == 2
        lidx = min(range(len(node.clades)), key=(lambda child : min(int(c.name) for c in node.clades[child].get_terminals())))
        name = 0 if len(internal) == 0 else max(internal) + 1
        internal.add(name)
        left_names, left_edges = smallleft_dsf_inner(node.clades[lidx], internal, leaves)
        right_names, right_edges = smallleft_dsf_inner(node.clades[1 - lidx], internal, leaves)
        left_names.update(right_names)
        left_names.update({node.name : name})
        return left_names, left_edges | right_edges | {(left_names[node.name], left_names[node.clades[lidx].name]),
                                                       (left_names[node.name], left_names[node.clades[1 - lidx].name])}
        
        
class ProgressBar:

    def __init__(self, total, length, counter=0, verbose=False, decimals=1, fill=chr(9608), prefix = 'Progress:', suffix = 'Complete'):
        self.total = total
        self.length = length
        self.decimals = decimals
        self.fill = fill
        self.prefix = prefix
        self.suffix = suffix
        self.counter = counter
        self.verbose = verbose

    def progress(self, advance=True, msg=""):
        self.progress_unlocked(advance, msg)
        return True

    def progress_unlocked(self, advance, msg):
        flush = sys.stderr.flush
        write = sys.stderr.write
        if advance:
            self.counter += 1
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.counter / float(self.total)))
        filledLength = int(self.length * self.counter // self.total)
        bar = self.fill * filledLength + '-' * (self.length - filledLength)
        rewind = '\x1b[2K\r'
        result = '%s |%s| %s%% %s' % (self.prefix, bar, percent, self.suffix)
        msg = '[{:%Y-%b-%d %H:%M:%S}]'.format(datetime.datetime.now()) + msg
        if not self.verbose:
            toprint = rewind + result + " [%s]" % (msg)
        else:
            toprint = rewind + msg + "\n" + result
        write(toprint)
        flush()
        if self.counter == self.total:
            write("\n")
            flush()

