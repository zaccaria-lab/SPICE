"""Bootstrap tree topologies for SPICE when no fixed topology is given.

Each bootstrap replicate builds one tree:

1. Resample the genome: from the chromosomes whose copy numbers differ between leaves, draw the same number of
   chromosomes with replacement, then, within each drawn chromosome, draw its segments with replacement (kept in
   genome order).
2. Add a normal leaf, node 0, with diploid copy number at every position of every allele.
3. Compute the distance between every pair of leaves: the sum over chromosomes and alleles of the distance function
   bphamm (number of breakpoints whose copy number change differs between the two profiles), minimised over the WGD
   options below, plus the cost of that option.
4. Build a neighbour-joining tree from the distance matrix and root it on the normal leaf.
5. Name the internal nodes 0, 1, 2, ... depth first, visiting the child containing the smallest leaf ID first, and
   drop the normal leaf, so the root is node 0.

WGD options
-----------
Add 0, 1 or 2 copies at every position (cost 0, 1 or 2 times `wgd_cost`), or double the profile ('x2', cost
`wgd_cost`). `wgd_cost` is `max_cn` if `weight_wgd` is True, otherwise 1.

The replicates run in a `multiprocessing.Pool`. `init_trees` copies the shared inputs into module-level globals in
each worker, and each replicate seeds NumPy with its own random job number, drawn from `rs`.

`tree_bootstrapping` returns one row per replicate: `tree_id` (the replicate's job number + 1), `tree` (sorted tuple
of (parent, child) edges), `dm` (lower-triangular distance matrix, normal leaf first) and `topology_id` (1 for the
most frequent topology, 2 for the next, ...).
"""

import itertools
from multiprocessing import Pool
from typing import Union

import numpy as np
import pandas as pd
from Bio.Phylo.BaseTree import Clade, Tree
from Bio.Phylo.TreeConstruction import DistanceMatrix as PDistanceMatrix
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor

from spice.libs.utils.progress import ProgressBar


def check_cn_same(profiles: dict, leaves: list, c: int, a: int) -> bool:
    """Check whether all leaves have the same copy numbers on one chromosome allele.

    Parameters
    ----------
    profiles : dict
        Leaf copy number profiles in the format {leaf: {chr: {allele: [cn per segment]}}}.
    leaves : list
        Leaves to compare; each is compared with the first.
    c : int
        Chromosome.
    a : int
        Allele (0 or 1).

    Returns
    -------
    bool
        True if every leaf has the same copy number as the first leaf at every segment.
    """
    return np.all([np.equal(profiles[leaves[0]][c][a], profiles[l][c][a]) for l in leaves[1:]])


def tree_bootstrapping(n_runs: int, leaves: dict, allele_specific: bool, weight_wgd: bool,
                       max_cn: int, rs: int, njobs: int = 1) -> pd.DataFrame:
    """Build bootstrap neighbour-joining trees and number their topologies by frequency.

    Only chromosomes whose copy numbers differ between leaves are resampled. Each replicate is built by
    `generate_tree` in a worker of a `multiprocessing.Pool`, seeded with its own random job number.

    Parameters
    ----------
    n_runs : int
        Number of bootstrap replicates. Must be greater than 0.
    leaves : dict
        Leaf copy number profiles in the format {leaf: {chr: {allele: [cn per segment]}}}.
    allele_specific : bool
        If True, each allele's copy number is modelled separately; if False, total copy number is modelled. Sets the
        number of alleles checked when finding the chromosomes that differ between leaves.
    weight_wgd : bool
        If True, a WGD costs `max_cn` in the distance; otherwise it costs 1.
    max_cn : int
        Maximum copy number, used as the WGD cost when `weight_wgd` is True.
    rs : int
        Random seed used to draw the job numbers that seed each replicate.
    njobs : int, optional
        Number of worker processes (default: 1).

    Returns
    -------
    pd.DataFrame
        One row per replicate, sorted by `topology_id`: `tree_id` (the replicate's job number + 1), `tree` (sorted
        tuple of (parent, child) edges, root 0), `dm` (lower-triangular distance matrix as a DataFrame, normal leaf
        first) and `topology_id` (1 for the most frequent topology, 2 for the next, ...).
    """
    assert n_runs > 0, ('The n_bootstraps parameter is set to 0, increase to run bootstrapping or provide a fixed '
                        'topology.')
    np.random.seed(rs)
    chrs = list(leaves[list(leaves.keys())[0]].keys())
    use_chrs = [c for c in chrs
                if not np.all([check_cn_same(leaves, list(leaves.keys()), c, a) for a in range(1 + allele_specific)])]
    assert len(use_chrs) > 0, ('All leaves have the same copy numbers on every chromosome, so there is nothing to '
                               'bootstrap')
    with Pool(processes=njobs, initializer=init_trees,
              initargs=(leaves, use_chrs, max_cn, allele_specific, weight_wgd)) as pool:
        jobs = np.random.choice(2**20, n_runs, replace=False)
        bar = ProgressBar(total=len(jobs), length=30, verbose=False)
        progress = (lambda e: bar.progress(advance=True, msg="Pair: {}".format(e)))
        bar.progress(advance=False, msg="Started")
        results = {i + 1: [tree, dm] for i, tree, dm in pool.imap_unordered(get_tree, jobs) if progress(i)}
    results_df = pd.DataFrame([results]).T.reset_index().rename(columns={0: 'tree', 'index': 'tree_id'})
    results_df[['tree', 'dm']] = pd.DataFrame(results_df['tree'].tolist(), index=results_df.index)
    unique_trees = {j: i + 1 for i, j in enumerate(list(results_df['tree'].value_counts().index))}
    results_df['topology_id'] = results_df['tree'].apply(lambda x: unique_trees[x])
    return results_df.sort_values(by='topology_id')


def get_tree(job: int) -> tuple[int, tuple, pd.DataFrame]:
    """Build one bootstrap tree in a worker process.

    Seeds NumPy with `job`, then calls `generate_tree` with the inputs that `init_trees` stored in the worker's
    module-level globals.

    Parameters
    ----------
    job : int
        Job number of the replicate, used as its random seed.

    Returns
    -------
    job : int
        The job number, unchanged.
    tree : tuple of tuple
        Sorted (parent, child) edges of the bootstrap tree, root 0.
    dm : pd.DataFrame
        Lower-triangular distance matrix of the replicate (NaN above the diagonal), normal leaf first.
    """
    np.random.seed(job)
    tree, dm = generate_tree(leaves=LEAVES, use_chrs=USE_CHRS, allele_specific=ALLELE_SPECIFIC, weight_wgd=WEIGHT_WGD,
                             max_cn=MAX_CN)
    return job, tree, dm


def init_trees(_leaves: dict, _use_chrs: list, _max_cn: int, _allele_specific: bool, _weight_wgd: bool) -> None:
    """Store the shared bootstrap inputs in module-level globals of a worker process.

    Used as the `multiprocessing.Pool` initializer in `tree_bootstrapping`, so the inputs are sent to each worker
    once rather than with every job; `get_tree` reads them.

    Parameters
    ----------
    _leaves : dict
        Leaf copy number profiles in the format {leaf: {chr: {allele: [cn per segment]}}}.
    _use_chrs : list
        Chromosomes whose copy numbers differ between leaves; only these are resampled.
    _max_cn : int
        Maximum copy number, used as the WGD cost when `_weight_wgd` is True.
    _allele_specific : bool
        If True, each allele's copy number is modelled separately; if False, total copy number is modelled.
    _weight_wgd : bool
        If True, a WGD costs `_max_cn` in the distance; otherwise it costs 1.
    """
    global LEAVES, USE_CHRS, MAX_CN, ALLELE_SPECIFIC, WEIGHT_WGD
    LEAVES = _leaves
    USE_CHRS = _use_chrs
    MAX_CN = _max_cn
    ALLELE_SPECIFIC = _allele_specific
    WEIGHT_WGD = _weight_wgd


def generate_tree(leaves: dict, use_chrs: list, allele_specific: bool, weight_wgd: bool,
                  max_cn: int) -> tuple[tuple, pd.DataFrame]:
    """Build one bootstrap neighbour-joining tree from resampled leaf profiles.

    Resamples the genome (`generate_bootstrap_leaves`), adds the normal leaf 0 (`add_fake_normal`), computes the
    pairwise distances (`get_distance_matrix`), builds a neighbour-joining tree rooted on the normal leaf
    (`make_neighbour_joining_tree`) and renames its nodes (`smallleft_dfs`). Uses NumPy's global random state, which
    `get_tree` seeds.

    Parameters
    ----------
    leaves : dict
        Leaf copy number profiles in the format {leaf: {chr: {allele: [cn per segment]}}}.
    use_chrs : list
        Chromosomes whose copy numbers differ between leaves; only these are resampled.
    allele_specific : bool
        If True, each allele's copy number is modelled separately; if False, total copy number is modelled. Sets the
        copy number of the normal leaf (`add_fake_normal`).
    weight_wgd : bool
        If True, a WGD costs `max_cn` in the distance; otherwise it costs 1.
    max_cn : int
        Maximum copy number, used as the WGD cost when `weight_wgd` is True.

    Returns
    -------
    tree : tuple of tuple
        Sorted (parent, child) edges of the bootstrap tree, root 0, without the normal leaf.
    dm : pd.DataFrame
        Lower-triangular distance matrix (NaN above the diagonal), with rows and columns in sorted node order: the
        normal leaf 0 first, then the leaves.
    """
    leaves_new = generate_bootstrap_leaves(leaves, use_chrs)
    leaves_new, leaf_names_normal = add_fake_normal(leaves_new, allele_specific)
    wgd_cost = max_cn if weight_wgd else 1
    distance_matrix, leaf_order = get_distance_matrix(leaves_new, wgd_cost)
    assert leaf_names_normal == leaf_order, ('The order of the leaf names in the distance matrix does not match the '
                                             'expected order.')
    T = make_neighbour_joining_tree(distance_matrix, leaf_names_normal)
    nj_edges = smallleft_dfs(T)
    return tuple(sorted(nj_edges)), pd.DataFrame(distance_matrix)


def generate_bootstrap_leaves(leaves: dict, use_chrs: list) -> dict:
    """Resample the genome of every leaf for one bootstrap replicate.

    Draws `len(use_chrs)` chromosomes with replacement from `use_chrs`. For each drawn chromosome, draws as many
    segments as it has, with replacement, and sorts them so they stay in genome order. The same draw is applied to
    every leaf and allele. The number of segments is taken from allele 0 of the first leaf, so all leaves and alleles
    must have the same segments.

    Parameters
    ----------
    leaves : dict
        Leaf copy number profiles in the format {leaf: {chr: {allele: [cn per segment]}}}.
    use_chrs : list
        Chromosomes whose copy numbers differ between leaves; only these are resampled.

    Returns
    -------
    dict
        Resampled profiles in the format {leaf: {chr: [[cn per segment] per allele]}}, with the drawn chromosomes
        numbered from 1 in the order drawn (a chromosome drawn twice appears twice, with different segments).
    """
    sampled_chrs = np.random.choice(use_chrs, len(use_chrs), replace=True)
    sample_segs = (lambda n: list(np.sort(np.random.choice(n, n, replace=True))))
    use_leaf = list(leaves.keys())[0]
    sampled_segs = {i: sample_segs(len(leaves[use_leaf][c][0])) for i, c in enumerate(sampled_chrs)}
    sampled_leaves = {l: {i + 1: [list(np.array(leaves[l][c][a])[sampled_segs[i]]) for a in range(len(leaves[l][c]))]
                          for i, c in enumerate(sampled_chrs)}
                      for l in leaves.keys()}
    return sampled_leaves


def add_fake_normal(leaves: dict, allele_specific: bool) -> tuple[dict, list]:
    """Add a diploid normal leaf, node 0, to resampled leaf profiles.

    The normal leaf has copy number 1 at every segment of each allele if `allele_specific` is True, or total copy
    number 2 at every segment if it is False, with the same chromosomes and segment counts as the first leaf. The
    tree is later rooted on it.

    Parameters
    ----------
    leaves : dict
        Resampled profiles in the format {leaf: {chr: [[cn per segment] per allele]}} (modified in place).
    allele_specific : bool
        If True, each allele's copy number is modelled separately; if False, total copy number is modelled.

    Returns
    -------
    leaves : dict
        The same dict, with the normal leaf 0 added.
    leaf_names_normal : list
        Sorted node IDs of all leaves, including 0.
    """
    leaf_names = list(leaves.keys())
    leaf_names_normal = [0] + leaf_names
    if not allele_specific:
        n_alleles = {len(cn) for profile in leaves.values() for cn in profile.values()}
        assert n_alleles == {1}, ('In total copy number mode (allele_specific=False) each chromosome must have one '
                                  f'allele, found {sorted(n_alleles)} alleles')
    normal_cn = 1 if allele_specific else 2
    leaves[0] = {ch: [list(np.full(len(i), normal_cn, dtype=int)) for i in cn]
                 for ch, cn in leaves[leaf_names[0]].items()}
    return leaves, sorted(leaf_names_normal)


# Apply a WGD option: add a number of copies at every segment, or add the profile itself to double it
add_wgd_effect = (lambda cell, wgd_add: cell + wgd_add)
# Pad a profile with copy number 1 at both chromosome ends (telomeres), so changes at the ends count as breakpoints
add_telomeres = (lambda cell: np.concatenate(([1], np.array(cell), [1])))
# Count the breakpoints at which two padded profiles change by different amounts
calc_bp_hammdist = (lambda cell_1, cell_2: np.sum(np.abs(np.diff(cell_1) != np.diff(cell_2))))


def get_breakpoint_distance(cell_1: list, cell_2: list, wgd_add: Union[int, str] = 0) -> float:
    """Breakpoint distance between two copy number profiles of one chromosome allele, after a WGD option.

    Applies the WGD option to one profile, pads both with copy number 1 at the ends (telomeres) and counts the
    breakpoints whose copy number change differs (`calc_bp_hammdist`). This is done twice, applying the WGD option to
    each profile in turn, and the smaller count is used. The number of breakpoints is halved such that the distance
    instead represents the number of interval events as opposed to the number of individual breakpoints (i.e. each
    event has two breakpoints).

    Parameters
    ----------
    cell_1, cell_2 : list
        Copy number per segment of the same chromosome allele in two cells.
    wgd_add : int or str, optional
        WGD option: the number of copies to add at every segment (default: 0), or 'x2' to double the profile.

    Returns
    -------
    float
        Half the smaller number of differing breakpoints.
    """
    cell_1 = np.array(cell_1)
    cell_2 = np.array(cell_2)
    wgd_add_c1 = wgd_add if isinstance(wgd_add, int) else cell_1
    wgd_add_c2 = wgd_add if isinstance(wgd_add, int) else cell_2

    d1 = calc_bp_hammdist(add_telomeres(cell_2), add_telomeres(add_wgd_effect(cell_1, wgd_add_c1)))
    d2 = calc_bp_hammdist(add_telomeres(cell_1), add_telomeres(add_wgd_effect(cell_2, wgd_add_c2)))
    return np.min([d1, d2]) / 2


def calculate_distance(leaves: dict, cell_1: int, cell_2: int, wgd_cost: int) -> float:
    """Distance between two cells: the breakpoint distance summed over the genome, minimised over WGD options.

    For each WGD option (add 0, 1 or 2 copies at every segment, or double the profile, 'x2'), sums
    `get_breakpoint_distance` over all chromosomes and alleles and adds the option's cost: the number of copies added
    times `wgd_cost`, or `wgd_cost` for 'x2'. Returns the smallest total.

    Parameters
    ----------
    leaves : dict
        Resampled profiles, including the normal leaf 0, in the format {leaf: {chr: [[cn per segment] per allele]}}.
    cell_1, cell_2 : int
        Node IDs of the two cells.
    wgd_cost : int
        Cost of one WGD: `max_cn` if `weight_wgd` is True, otherwise 1.

    Returns
    -------
    float
        The smallest total distance over the WGD options.
    """
    check_profiles_same_len = (lambda leaves, cell_1, cell_2:
                               np.all([len(leaves[cell_1][ch][allele]) == len(leaves[cell_2][ch][allele])
                                       for ch, profile in leaves[cell_1].items()
                                       for allele in range(len(profile))]))
    assert (leaves[cell_1].keys() == leaves[cell_2].keys()) and (check_profiles_same_len(leaves, cell_1, cell_2)), \
        f'Cells {cell_1} and {cell_2} must have the same chromosomes and number of segments per allele'
    get_wgd_add_value = (lambda x: x if isinstance(x, int) else 1)
    res = [sum(get_breakpoint_distance(leaves[cell_1][ch][a], leaves[cell_2][ch][a], wgd_add)
               for ch, profile in leaves[cell_1].items()
               for a in range(len(profile))) + (get_wgd_add_value(wgd_add) * wgd_cost)
           for wgd_add in [0, 1, 2, 'x2']]
    min_res = min(res)
    return min_res


def get_distance_matrix(leaves: dict, wgd_cost: int) -> tuple[list, list]:
    """Compute the distance between every pair of cells, as a lower-triangular matrix.

    Parameters
    ----------
    leaves : dict
        Resampled profiles, including the normal leaf 0, in the format {leaf: {chr: [[cn per segment] per allele]}}.
    wgd_cost : int
        Cost of one WGD: `max_cn` if `weight_wgd` is True, otherwise 1.

    Returns
    -------
    dm : list of list
        Lower-triangular distance matrix in sorted node order, with 0 on the diagonal: row k holds the distances from
        the k-th node to the nodes before it, then 0 (the format of `Bio.Phylo`'s `DistanceMatrix`).
    nodes : list
        Sorted node IDs, in the order of the rows of `dm`.
    """
    pairs = set((i, j) if i >= j else (j, i) for i, j in itertools.combinations(leaves.keys(), 2))
    distance_matrix = {pair: calculate_distance(leaves, pair[0], pair[1], wgd_cost) for pair in pairs}
    dm = [[distance_matrix[j, i] for i in sorted(leaves.keys()) if j > i] + [0] for j in sorted(leaves.keys())]
    return dm, sorted(leaves.keys())


def make_neighbour_joining_tree(distance_matrix: list, leaves: list) -> Tree:
    """Build a neighbour-joining tree with Biopython and root it on the normal leaf 0.

    Parameters
    ----------
    distance_matrix : list of list
        Lower-triangular distance matrix with 0 on the diagonal, from `get_distance_matrix`.
    leaves : list
        Node IDs of the rows of `distance_matrix`, including the normal leaf 0.

    Returns
    -------
    Bio.Phylo.BaseTree.Tree
        The rooted tree. Leaf clades are named by node ID as strings; internal clades have Biopython's names
        ('Inner1', ...), and the root has no name.
    """
    T = DistanceTreeConstructor().nj(PDistanceMatrix(names=list(map(str, leaves)), matrix=distance_matrix))
    T.root_with_outgroup('0')
    return T


def smallleft_dfs(T: Tree) -> list:
    """Rename the nodes of a rooted tree to integer IDs and list its edges, without the normal leaf.

    Internal nodes are numbered 0, 1, 2, ... depth first from the root, each before its children, visiting first the
    child that contains the smallest leaf ID (`smallleft_dfs_inner`). Leaves keep their node IDs. The root therefore
    gets 0, the ID of the normal leaf, and the edge from the root to the normal leaf is dropped.

    Parameters
    ----------
    T : Bio.Phylo.BaseTree.Tree
        Rooted tree from `make_neighbour_joining_tree`, with leaves named by node ID.

    Returns
    -------
    list of tuple
        (parent, child) edges, root 0, without the edge to the normal leaf, in no particular order.
    """
    _, edges = smallleft_dfs_inner(T.root, set())
    nj_edges = [(i, j) for i, j in edges if str(j) != '0']
    return nj_edges


def smallleft_dfs_inner(node: Clade, internal: set) -> tuple[dict, set]:
    """Recursively name the internal nodes below `node` and collect the edges of its subtree.

    A leaf keeps its node ID (its name as an integer). An internal node gets the next free ID (one above the largest
    in `internal`, or 0 for the first) before its children are visited, and the child that contains the smallest leaf
    ID is visited first.

    Parameters
    ----------
    node : Bio.Phylo.BaseTree.Clade
        Root of the subtree. Each internal node must have exactly two children.
    internal : set
        IDs already given to internal nodes (modified in place).

    Returns
    -------
    names : dict
        Integer node ID for each clade name in the subtree (the unnamed root of the tree is keyed None).
    edges : set of tuple
        (parent, child) edges of the subtree, as node IDs.
    """
    if node.is_terminal():
        return {node.name: int(node.name)}, set()
    else:
        assert len(node.clades) == 2, f'The tree must be binary: clade {node.name} has {len(node.clades)} children'
        lidx = min(range(len(node.clades)),
                   key=(lambda child: min(int(c.name) for c in node.clades[child].get_terminals())))
        name = 0 if len(internal) == 0 else max(internal) + 1
        internal.add(name)
        left_names, left_edges = smallleft_dfs_inner(node.clades[lidx], internal)
        right_names, right_edges = smallleft_dfs_inner(node.clades[1 - lidx], internal)
        left_names.update(right_names)
        left_names.update({node.name: name})
        return left_names, left_edges | right_edges | {(left_names[node.name], left_names[node.clades[lidx].name]),
                                                       (left_names[node.name], left_names[node.clades[1 - lidx].name])}
