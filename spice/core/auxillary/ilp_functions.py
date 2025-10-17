import numpy as np
import pandas as pd
import pyomo.environ as pe

from spice.core.auxillary.edges_functions import *

def check_leaf_input(leaves, allele_specific, max_cn, logger):
    test_leaf = list(leaves.keys())[0]
    test_chr = list(leaves[test_leaf].keys())[0]
    assert (type(leaves) == dict) & (type(leaves[test_leaf]) == dict) & (type(leaves[test_leaf][test_chr]) == dict), "Leaves must be a dictionary in the format {leaf: {chr: {allele : [cn]}}}"
    check_allele_specific = (lambda p, allele_specific : len(p) == allele_specific + 1)
    assert all(check_allele_specific(cns, allele_specific) for leaf, profiles in leaves.items() for ch, cns in profiles.items()), "If running in allele-specific mode, cn format must be {0 : [alleleA cns], 1 : [alleleB cns]} else cn format must be {0 : [total cns]}"
    # check_cn_length = (lambda p, compare : len(p) == len(compare)) #! CHANGE BELOW TO CHECK THAT CHR & ALLELES SAME SIZE FOR ALL LEAVES.
    # assert all(check_cn_length(allele_cns, leaves[test_leaf][ch][0]) for leaf, profiles in leaves.items() for ch, cns in profiles.items() for allele_cns in cns.values()), 'Length of copy number profiles in all leaves and alleles MUST match.'
    input_max_cn = np.max([np.max(allele_cns) for leaf, profiles in leaves.items() for ch, cns in profiles.items() for allele_cns in cns.values()])
    if input_max_cn > max_cn:
        leaves = {leaf: {ch: {allele: np.clip(allele_cns, a_min=0, a_max=max_cn) for allele, allele_cns in cns.items()} for ch, cns in profiles.items()} for leaf, profiles in leaves.items()}
        logger.warning('Input data maximum CN ({}) exceeds specified maximum CN ({}). Clipping CN values to {}.'.format(input_max_cn, max_cn, max_cn))
    return leaves

def get_seg_pairs(segs, leaves, ch, allele, single_seg_chrs, seg_codes, logger, dir_spec=True):
    seg_pairs = [(p, q) for p in segs for q in segs if p <= q]
    use_code = 0
    original_n_segs = len(seg_pairs)
    if original_n_segs > 1000:
        # profiles = leaves_to_array(leaves, ch, allele)
        # seg_pairs = get_breakpoint_combinations(profiles, direction_specific=dir_spec)
        # use_code = 1
        # if len(seg_pairs) > 1000:
        seg_pairs = [(p, p) for p in segs]
        use_code = 2
        single_seg_chrs.append((ch, allele))
    if use_code > 0:
        code_msg = 'pairs of segments between leaf breakpoints' if use_code == 1 else 'single segments'
        msg = f'Number of segment pairs for chromosome {ch} allele {allele} ({original_n_segs}) exceeds SPICE capacity (1000), using {code_msg} instead - new number of segment pairs: {len(seg_pairs)}'
        logger.warning(msg)
    seg_codes.append(use_code)
    return seg_pairs

def get_breakpoint_combinations(profiles, direction_specific=False):
    profile_pad = np.pad(profiles, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=1)
    profile_shift = np.pad(profiles, pad_width=((0, 0), (1, 0)), mode='constant', constant_values=1)
    diff = profile_pad - profile_shift
    if direction_specific:
        pos_bp = np.argwhere(np.any(diff > 0, axis=0).astype(int)).flatten()+1
        neg_bp = np.argwhere(np.any(diff < 0, axis=0).astype(int)).flatten()
        seg_pairs = sorted(list({(i, j) for i in pos_bp for j in neg_bp if i <= j} | {(j, i) for i in pos_bp for j in neg_bp if j <= i}))
    else:
        any_bp = np.argwhere(np.any(diff[:-1] != 0, axis=0).astype(int)).flatten()+1
        seg_pairs = [(i, j) for i in any_bp for j in any_bp if i <= j]
    return seg_pairs

leaves_to_array = (lambda leaves, ch, allele : np.vstack([profile[ch][allele] for l, profile in leaves.items()]))


def process_tree_edges(leaves, nj_edges, topology, rs):

    n_leaves = len(leaves)

    if nj_edges is not None:
        edges = nj_edges
        topology = False
    else:
        if topology == True:
            edges = [(0, 1)] + [(i, j) for i in range(1, n_leaves) for j in range(i+1, 2*n_leaves)] 
        elif type(topology) == list:
            edges = topology
            topology = False
        else:
            edges = get_fixed_edges(leaves, rs)

    return edges, topology


def get_model_variables(model):
    if model is None:
        return []
    else:
        block_vars = set([var.parent_component() for var in model.component_data_objects(pe.Var)])
        dc = [str(bv).split('.')[-1] for bv in block_vars]
        return dc
    
get_n_modelvars = (lambda m: sum(len(list(var.values())) for var in list(m.component_map(ctype=pe.Var).values())))
get_n_modelcons = (lambda m: sum(len(list(con.values())) for con in list(m.component_map(ctype=pe.Constraint).values())))


get_gap = (lambda results: np.round((int(results['Problem'][0]['Upper bound']) - int(results['Problem'][0]['Lower bound']))*100/int(results['Problem'][0]['Upper bound']), 2))

def get_solution_objective(opt, sol):
    opt._solver_model.params.SolutionNumber = sol
    return np.round(opt._solver_model.getAttr('PoolObjVal'), 0)

def get_solution_status(opt):
    status_code = {2 : 'optimal', 3: 'infeasible', 8 : 'node_limit', 9 : 'time_limit', 10 : 'solution_limit', 11: 'user_interrupt'}
    return status_code[opt._solver_model.getAttr('Status')] if opt._solver_model.getAttr('Status') in [2, 3, 8, 9, 10, 11] else np.nan


def check_wgd_presence(nodes_data, edges, model, seg_sizes, mapping, threshold, logger):
    nodes = nodes_data.copy()
    mapping = {k : mapping[k] if k in mapping.keys() else 'internal_' + str(k) for k in nodes['Node'].unique()}
    parents, children = edges[edges.WGD > 0][['Parent', 'Child']].values.T
    nodes['Allele'] = nodes.Allele.astype(str).str.replace('0', 'A').str.replace('1', 'B')
    nodes['Pos'] = nodes['Pos'].astype(float)
    nodes_table = nodes.pivot_table(index='Node', columns=['Allele', 'Chr', 'Pos'], values='CN')
    diff = pd.DataFrame((nodes_table.reindex(children).values >= (2 * nodes_table.reindex(parents)).values) | (nodes_table.reindex(children).values == 0), index=children, columns=nodes_table.columns).T
    diff_sizes = diff.join(seg_sizes)
    diff_sizes[children] = diff_sizes[children].multiply(diff_sizes['segsize'], axis=0)
    diff_sizes = diff_sizes.reset_index().groupby('Allele').sum()
    prop_true = diff_sizes[children].divide(diff_sizes['segsize'], axis=0)
    wgd_present = (prop_true > threshold).all(axis=0).astype(int)
    edges['WGD'] = (edges['WGD'] * edges['Child'].map(wgd_present).fillna(1)).astype(int)
    for node in wgd_present[wgd_present == 0].index:
        logger.info(f'WGD is called by the model for the branch incoming to node {mapping[node]} however less than {threshold*100}% of each allele is doubled (A: {round(prop_true[node]["A"]*100)}%, B: {round(prop_true[node]["B"]*100)}%). Manually correcting WGD to not present.')
        model.w[node] = 0
    return edges, model



