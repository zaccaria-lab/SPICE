import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager
import re
import datetime
import gc

from scipy.spatial.distance import pdist, cdist, squareform
from Bio.Phylo.BaseTree import Tree, Clade
from Bio import Phylo

def create_pseudo_allelespecific(df):
    assert (len(df['Allele'].unique()) == 1)
    df_new_allele = df.copy()
    cn_col = [i for i in df.columns if 'cn' in i.lower()]
    for i in cn_col:
        df_new_allele[i] = 0
    events_col = [i for i in df.columns if 'del' in i.lower()] + [i for i in df.columns if 'gain' in i.lower()]
    for i in events_col:
        df_new_allele[i] = 0
    new_allele = 'B' if df['Allele'].unique()[0] == 'A' else 1 
    df_new_allele['Allele'] = new_allele
    return pd.concat([df, df_new_allele]).reset_index(drop=True)

def fill_removed_segments(total_nodes, total_edges, all_leaves, final_mapping, maxcn, j, focal_maxcn=None, logger=None, 
                          allele_specific=True, to_phase=False):
    final_mapping['Allele'] = final_mapping['Allele'].map({'A' : 0, 'B' : 1})
    focal_maxcn = maxcn if focal_maxcn is None else focal_maxcn
    if allele_specific == False:
        total_nodes_use = create_pseudo_allelespecific(total_nodes)
        all_leaves_use = create_pseudo_allelespecific(all_leaves)
        final_mapping_use = create_pseudo_allelespecific(final_mapping)
    else:
        total_nodes_use = total_nodes
        all_leaves_use = all_leaves
        final_mapping_use = final_mapping
    if logger is None:
        logger, logfile = initialize_logger(True, None)
        close = True
    else:
        close = False    
    logger.trace('Running segment tree mapping')
    pos_locs = all_leaves_use[['Chr', 'Pos', 'Start', 'End']].drop_duplicates()

    all_leaves_use = all_leaves_use.drop(['Start', 'End'], axis=1)
    output, states, neighbour_states, total_nodes_mergepos = get_unique_state_trees(total_nodes_use, total_edges, 
                                                                                    all_leaves_use, final_mapping_use, 
                                                                                    maxcn, j, logger)
    filled_profiles = multi_fill_gaps(states, neighbour_states, j, focal_maxcn=focal_maxcn, weighted=True, 
                                      allspec=allele_specific)
    res = (output
           .merge(filled_profiles, on='state_tree_id', how='left')
           [['Solution', 'Node', 'Chr', 'Pos', 'cnA', 'cnB', 'DeletionsA', 'DeletionsB', 'GainsA', 'GainsB']]
           .astype(int, errors='ignore'))
    final = (pd.concat([process_nodes_output(total_nodes_mergepos).assign(method='spice'), res.assign(method='fillgaps')])
             .merge(pos_locs, on=['Chr', 'Pos'])
             .sort_values(by=['Solution', 'Node', 'Chr', 'Pos'])
             [['Solution', 'Node', 'Chr', 'Pos', 'Start', 'End', 'cnA', 'cnB', 'DeletionsA', 'DeletionsB', 'GainsA', 
               'GainsB', 'method']])
    final.reset_index(drop=True, inplace=True)
    if allele_specific == False:
        res = (res[['Solution', 'Node', 'Chr', 'Pos', 'cnA', 'DeletionsA', 'GainsA']]
               .rename(columns={'cnA' : 'cn', 'DeletionsA' : 'Deletions', 'GainsA' : 'Gains'}))
        final = (final[['Solution', 'Node', 'Chr', 'Pos', 'Start', 'End', 'cnA', 'DeletionsA', 'GainsA', 'method']]
                 .rename(columns={'cnA' : 'cn', 'DeletionsA' : 'Deletions', 'GainsA' : 'Gains'}))
    total_edges = pd.concat([total_edges, 
                             add_focal_events_to_edgesdf(res, total_edges),
                             add_focal_events_to_edgesdf(res, total_edges, interval_focal=True, final=final)], axis=1)
    if close:
        logger.removeHandler(logfile)
        logger.handlers.clear()
        
    return final, total_edges

def process_nodes_output(total_nodes):
    total_nodes['Pos'] = total_nodes['Pos'].astype(int)
    total_nodes['Allele'] = total_nodes['Allele'].map({0 : 'A', 1 : 'B'})
    idx_cols = [i for i in total_nodes.columns if i not in ['Allele', 'CN', 'Gains', 'Deletions']]
    total_nodes = total_nodes.rename(columns={'CN' : 'cn'}).pivot(index=idx_cols, columns=['Allele'])
    total_nodes.columns = [''.join(col) for col in total_nodes.columns.values]
    total_nodes.reset_index(inplace=True)
    return total_nodes


def get_unique_state_trees(total_nodes, total_edges, all_leaves, final_mapping, maxcn, j, logger=None):
    removed_leaves = all_leaves[all_leaves.filter_seg].copy()
    kept_leaves = all_leaves[~all_leaves.filter_seg].copy()
    logger.trace('Getting neighbour CN profiles') 
    leaf_distances = get_removed_pos_closest_neighbour(removed_leaves, kept_leaves, maxcn)
    leaf_profiles = get_leaf_node_states(removed_leaves, total_edges)
    total_nodes = total_nodes.merge(final_mapping[['Chr', 'Allele', 'finalPos', 'mergePos']].drop_duplicates(), 
                                    left_on=['Chr', 'Allele', 'Pos'], right_on=['Chr', 'Allele', 'finalPos'])
    logger.trace('Getting branch CN states') 
    internal_profiles = multi_get_internal_states(total_nodes, total_edges, leaf_distances, final_mapping, j)
    neighbour_profiles = internal_profiles.copy()
    leaf_nodes = all_leaves['Node'].unique()
    total_profiles = pd.concat([internal_profiles[~internal_profiles.Child.isin(leaf_nodes)], leaf_profiles])
    logger.trace('Getting topology CN states') 
    neighbour_profiles = (neighbour_profiles
                          .groupby(['Solution', 'Chr', 'Pos'])['Branch_state']
                          .apply(lambda x: tuple(np.sort(x.values)))
                          .sort_values()
                          .reset_index()
                          .rename(columns={'Branch_state' : 'Neighbour_state'}))
    total_profiles = (total_profiles
                      .groupby(['Solution', 'Chr', 'Pos'])['Branch_state']
                      .apply(lambda x: tuple(np.sort(x.values)))
                      .sort_values()
                      .reset_index())
    total_profiles = total_profiles.merge(neighbour_profiles, on=['Solution', 'Chr', 'Pos'], how='outer')
    assert total_profiles.isnull().any(axis=1).sum() == 0
    total_profiles['state_tree_id'] = ((total_profiles['Branch_state'] != total_profiles['Branch_state'].shift(1)) | 
                                       (total_profiles['Neighbour_state'] != total_profiles['Neighbour_state'].shift(1))).cumsum()
    return (total_profiles[['Solution', 'Chr', 'Pos', 'state_tree_id']], 
            total_profiles.drop_duplicates(subset=['state_tree_id']).set_index('state_tree_id')['Branch_state'],
            total_profiles.drop_duplicates(subset=['state_tree_id']).set_index('state_tree_id')['Neighbour_state'], 
            total_nodes.drop(['Pos', 'finalPos', 'split_group'], axis=1).rename(columns={'mergePos' : 'Pos'}))

def get_removed_pos_closest_neighbour(removed_leaves, kept_leaves, maxcn):
    pos_neighbours = (removed_leaves[['Chr', 'Pos']]
                      .drop_duplicates()
                      .groupby('Chr')['Pos']
                      .apply(lambda x: get_neighbours(x.values))
                      .reset_index())
    kept_profiles = (kept_leaves
                     .merge(pos_neighbours[['Chr', 'neighbours']], left_on=['Chr', 'Pos'], 
                            right_on=['Chr', 'neighbours'], how='inner')
                     .rename(columns={'Chr' : 'Chr_k'})
                     .pivot_table(index=['Chr_k', 'neighbours'], columns=['Node', 'Allele'], values='CN'))
    removed_profiles = (removed_leaves
                        .merge(pos_neighbours[['Chr', 'removed_pos']], left_on=['Chr', 'Pos'], 
                               right_on=['Chr', 'removed_pos'], how='inner')
                        .rename(columns={'Chr' : 'Chr_r'})
                        .pivot_table(index=['Chr_r', 'removed_pos'], columns=['Node', 'Allele'], values='CN'))
    result = pd.melt(pd.DataFrame(cdist(removed_profiles, kept_profiles, metric=compare_leaves, max_cn=maxcn), 
                                  index=removed_profiles.index, columns=kept_profiles.index), ignore_index=False).reset_index()
    result = result[result.Chr_r == result.Chr_k].rename(columns={'Chr_k' : 'Chr'}).drop(['Chr_r'], axis=1)
    result = (pos_neighbours
              .merge(result, on=['Chr', 'removed_pos', 'neighbours'], how='inner')
              .sort_values(by=['Chr', 'removed_pos', 'value'], ascending=True)
              .drop_duplicates(subset=['Chr', 'removed_pos'], keep='first'))
    return result

def compare_leaves(profile_1, profile_2, max_cn):
    a = np.sum(profile_1 != profile_2)
    b = np.sum(abs(profile_1 - profile_2))
    return a + (b/(max_cn*len(profile_1)))

def get_neighbours(positions):
    groups = np.split(positions, np.nonzero(np.diff(positions) != 1)[0] + 1)
    return (pd.DataFrame(list(zip(groups, [[np.min(pos)-1, np.max(pos)+1] for pos in groups])), columns=['removed_pos', 'neighbours'])
            .explode('removed_pos')
            .explode('neighbours')
            .set_index('removed_pos'))

def get_leaf_node_states(removed_leaves, total_edges):
    output = (removed_leaves
              .groupby(['Node', 'Chr', 'Pos'])['CN']
              .agg(lambda x: tuple(x.values.astype(int)))
              .reset_index()
              .rename(columns={'CN' : 'State'}))
    output = (output
              .merge(total_edges[['Solution', 'Parent', 'Child', 'WGD']], left_on=['Node'], right_on=['Child'], how='left')
              .set_index(['Solution', 'Chr', 'Pos'])[['Parent', 'Child', 'State', 'WGD']])
    output['Branch_state'] = list(map(tuple, output.values))
    return output['Branch_state'].reset_index()

def multi_get_internal_states(total_nodes, total_edges, leaf_distances, final_mapping, j):
    split_size = int(np.minimum(np.ceil(total_edges.Solution.nunique() / j), 2000))
    n_splits = int(np.ceil(total_edges.Solution.nunique() / split_size))
    splits = np.array_split(total_edges.Solution.unique(), n_splits)
    splits = {j : i for i in range(n_splits) for j in splits[i]}
    total_nodes['split_group'] = total_nodes['Solution'].map(splits)
    total_edges['split_group'] = total_edges['Solution'].map(splits)
    nodes_list = {k : v for k, v in total_nodes.groupby('split_group')}
    edges_list = {k : v for k, v in total_edges.groupby('split_group')}
    
    with Pool(processes=j, initializer=init_unique_trees, initargs=(nodes_list, edges_list, leaf_distances, final_mapping)) as pool:
        jobs = list(nodes_list.keys())
        bar = ProgressBar(total= len(jobs), length=30, verbose=False)
        progress = (lambda job, split_size : bar.progress(advance=True, msg="Solutions: {}-{}".format((job*split_size)+1, 
                                                                                                      (job*split_size) + (split_size))))
        output = [out for sols, out in pool.imap_unordered(get_job_internal_nodes, jobs) if progress(sols, split_size)]
        
    return pd.concat(output).reset_index(drop=True)

def init_unique_trees(_nodes_list, _edges_list, _leaf_distances, _final_mapping):
    global NODES_LIST, EDGES_LIST, LEAF_DISTANCES, FINAL_MAPPING 
    NODES_LIST = _nodes_list
    EDGES_LIST = _edges_list
    LEAF_DISTANCES = _leaf_distances
    FINAL_MAPPING = _final_mapping
    
def get_job_internal_nodes(job):
    internal_profiles = get_internal_node_states(NODES_LIST[job], EDGES_LIST[job], LEAF_DISTANCES, FINAL_MAPPING)
    return job, internal_profiles

def get_internal_node_states(total_nodes, total_edges, leaf_distances, final_mapping):
    use_nodes = total_nodes.merge(leaf_distances, left_on=['Chr', 'mergePos'], 
                                  right_on=['Chr', 'neighbours'])[['Solution', 'Node', 'Chr', 'removed_pos', 'Allele', 'CN']]
    use_nodes = use_nodes.merge(total_edges, left_on=['Solution', 'Node'], right_on=['Solution', 'Child'])
    use_nodes = use_nodes.pivot_table(index=['Solution', 'Chr', 'removed_pos', 'Parent', 'Child', 'WGD'], 
                                      columns='Allele', values='CN').astype(int)
    use_nodes['CN'] = list(map(tuple, use_nodes.values))
    use_nodes.reset_index(level=['Parent', 'Child', 'WGD'], inplace=True)
    use_nodes['Branch_state'] = list(map(tuple, use_nodes[['Parent', 'Child', 'CN', 'WGD']].values))
    use_nodes.reset_index(inplace=True)
    return use_nodes[['Solution', 'Chr', 'removed_pos', 'Child', 'Branch_state']].rename(columns={'removed_pos' : 'Pos'})

def multi_fill_gaps(states, neighbour_states, j, focal_maxcn, weighted=False, allspec=True):
    with Manager() as manager:
        inputs = manager.dict(states.to_dict())
        nb_inputs = manager.dict(neighbour_states.to_dict())
        with Pool(processes=j, 
                #   maxtasksperchild=1,
                  initargs=(inputs, nb_inputs, focal_maxcn, weighted, allspec),
                  initializer=init_fill_gaps) as pool:
            bar = ProgressBar(total=len(states), length=30, verbose=False)
            progress = (lambda e : bar.progress(advance=True, msg="Tree {}".format(e)))
            bar.progress(advance=False, msg="Started")
            results = [result_df.assign(state_tree_id=idx) for idx, result_df in pool.imap_unordered(parallel_fill_gaps, states.index) if progress(idx)]
    return pd.concat(results)

def init_fill_gaps(_inputs, _nb_inputs, _focal_maxcn, _weighted, _allspec):
    global INPUTS, NBINPUTS, FOCALMAXCN, WEIGHTED, ALLSPEC
    INPUTS = _inputs
    NBINPUTS = _nb_inputs
    FOCALMAXCN = _focal_maxcn
    WEIGHTED = _weighted
    ALLSPEC = _allspec

def parallel_fill_gaps(idx):
    result = fill_internal_nodes(INPUTS[idx], NBINPUTS[idx], FOCALMAXCN, WEIGHTED, ALLSPEC)
    return idx, result

def fill_internal_nodes(state_tree, nb_state_tree, focal_maxcn, weighted, allele_specific):
    parents, children, states, wgd = list(zip(*state_tree))
    _, nb_children, nb_states, _ = list(zip(*nb_state_tree))
    root = [i for i in parents if i not in children][0]
    edges = pd.DataFrame(list(zip(parents, children)), columns=['Parent', 'Child'])
    mrca = edges[edges.Parent == root]['Child'].unique()[0]
    edges = edges[edges.Parent != root]
    edges['Parent'] = edges['Parent'].astype(str)
    edges['Child'] = edges['Child'].astype(str)
    edges['n_events'] = 0
    state_mapping = dict(zip([str(i) for i in children], states))
    nb_state_mapping = dict(zip([str(i) for i in nb_children], nb_states))
    iswgd = dict(zip([str(i) for i in children], wgd))
    make_clade = (lambda node, edges, length=0 : Clade(name=node, 
                                                   clades=[make_clade(child['Child'], edges, child['n_events']) 
                                                          for _, child in edges[edges['Parent']==node].iterrows()],
                                                   branch_length=length)
                                              if node in edges['Parent'].to_list() else Clade(name=node, branch_length=length))
    
    T = Tree().from_clade(make_clade(mrca, edges))
    node_states, _, gains_losses = fill_gaps(T, state_mapping, nb_state_mapping, weighted=weighted, iswgd=iswgd, 
                                             max_cn_eval=focal_maxcn, is_all=allele_specific)
    D = pd.DataFrame(list(map(lambda x : dict(zip(('Node', 'cn', 'Deletions', 'Gains', 'WGD'), 
                                                  (x[0], node_states[x[0]]) + x[1])), gains_losses.items())))
    D.index = D.index + 1
    D['cnA'], D['cnB'] = zip(*D['cn'])
    D['DeletionsA'], D['DeletionsB'] = zip(*D['Deletions'])
    D['GainsA'], D['GainsB'] = zip(*D['Gains'])
    if allele_specific == True:
        D.loc[0] = [root, (1, 1), (0, 0), (0, 0), 0, 1, 1, 0, 0, 0, 0]
    else:
        D.loc[0] = [root, (2, 0), (0, 0), (0, 0), 0, 2, 0, 0, 0, 0, 0]
    return D[['Node', 'cnA', 'cnB', 'DeletionsA', 'DeletionsB', 'GainsA', 'GainsB']]
 
def fill_gaps(tree, incns, neighbours, iswgd=None, add_diploid_root=True, weighted=False, max_cn_eval=20, 
              to_phase=False, is_all=True): 
    assert (all(isinstance(v, np.int64) for v in incns.values()) or 
            all(isinstance(v[0], np.int64) and isinstance(v[1], np.int64) for v in incns.values())), 'Input_cns is not integer nor pairs of int'
    assert ('fixed-root' not in incns) or (not add_diploid_root), ''
    T, cns = (Tree(Clade(name='fixed-root', clades=[tree.clade])), 
              incns | {'fixed-root' : (1, 1) if is_all else (2, 0)}) if add_diploid_root else (tree, incns)
    input_cns = cns 
    save_leaves = {leaf.name : input_cns[leaf.name] for leaf in T.clade.get_terminals()}
    input_cns = {v : (min(input_cns[v][0], max_cn_eval), min(input_cns[v][1], max_cn_eval)) for v in input_cns}
    if add_diploid_root:
        neighbours = neighbours | {'fixed-root' : (1, 1) if is_all else (2, 0)}
    nb_cns = neighbours
    nb_cns = {v : (min(nb_cns[v][0], max_cn_eval), min(nb_cns[v][1], max_cn_eval)) for v in nb_cns}
    max_cn = max(v[0] + v[1] for v in input_cns.values())
    norm = ((max_cn + 1) * (sum(1 for _ in T.find_clades(terminal=False))) + 1) * 2
    calc_cost = (lambda v, u : (abs(v[0] - u[0]) + abs(v[1] - u[1]) if weighted else int((v[0] != u[0]) or (v[1] != u[1]))))
    norm_cost = (lambda cn, ch : (abs(cn[0] - ch[0]) + abs(cn[1] - ch[1])) / norm)

    same_dir = (lambda cnch_diff, cnichi_diff : ((cnch_diff >= 0) & (cnichi_diff >= 0)) | 
                                                ((cnch_diff < 0) & (cnichi_diff < 0)))
    calc_diff = (lambda cnch_diff, cnichi_diff : np.maximum(0, abs(cnch_diff) - (abs(cnichi_diff) * same_dir(cnch_diff, cnichi_diff))))
    
    new_cost_weighted = (lambda cn, ch, cni, chi : calc_diff(cn[0] - ch[0], cni[0] - chi[0]) +
                                                   calc_diff(cn[1] - ch[1], cni[1] - chi[1]))
    new_cost_unweighted = (lambda cn, ch, cni, chi : np.minimum(1, calc_diff(cn[0] - ch[0], cni[0] - chi[0])) +
                                                    np.minimum(1, calc_diff(cn[1] - ch[1], cni[1] - chi[1])))
    
    def new_cost(cn, ch, cni, chi, weighted=True): 
        if weighted:
            cost = new_cost_weighted(cn, ch, cni, chi)
        else:
            cost = new_cost_unweighted(cn, ch, cni, chi)
        return cost
    
    swap = (lambda P : (P[1], P[0]))
    if to_phase:
        cost = {leaf.name : {input_cns[leaf.name] : 0, swap(input_cns[leaf.name]) : 0} for leaf in T.clade.get_terminals()}
        back = {leaf.name : {input_cns[leaf.name] : None, swap(input_cns[leaf.name]) : None} for leaf in T.clade.get_terminals()}
    else:
        cost = {leaf.name : {input_cns[leaf.name] : 0} for leaf in T.clade.get_terminals()}
        back = {leaf.name : {input_cns[leaf.name] : None} for leaf in T.clade.get_terminals()}
    mkpairs = (lambda M : ((i, n - i) for n in range(M + 1) for i in range(n + 1)))
    it_cns = (lambda n : mkpairs(max_cn) if T.root != n else (input_cns[T.root.name], swap(input_cns[T.root.name])))
    check_loss = (lambda cn, cn_child : (cn[0] > 0 or cn_child[0] == 0) and (cn[1] > 0 or cn_child[1] == 0))

    for node in T.find_clades(order='postorder', terminal=False):
        opts = {cn : {child.name : {cn_child : (cost[child.name][cn_child] + new_cost(cn, cn_child, nb_cns[node.name], 
                                                                                      nb_cns[child.name], weighted=False)) 
                                    if check_loss(cn, cn_child) else np.inf 
                                    for cn_child in cost[child.name]}
                        for child in node.clades}
                for cn in it_cns(node)}
        
        cost[node.name] = {cn : sum(min(opts[cn][child].values()) for child in opts[cn]) for cn in opts}
        back[node.name] = {cn : {child : min(sorted(opts[cn][child].keys(), 
                                                    key=(lambda x : (abs(cn[0] - x[0]) + abs(cn[1] - x[1]), x[0] + x[1], x[0], x[1]))),
                                            key=(lambda cn_child : opts[cn][child][cn_child])) 
                                for child in opts[cn]}
                            for cn in opts}  
    assert not np.isnan(cost[T.root.name][input_cns[T.root.name]])

    backtrack = (lambda subroot, cn : set.union(*(backtrack(child, back[subroot][cn][child]) 
                                            for child in back[subroot][cn])).union({(subroot, cn)})
                                if back[subroot][cn] is not None else {(subroot, cn)})
    result = dict(backtrack(T.root.name, input_cns[T.root.name]))
    if to_phase:
        choose_leaf = (lambda inf, inp : min((inp, swap(inp)), key=(lambda x : abs(x[0] - inf[0]) + abs(x[1] - inf[1]))))
    else:
        choose_leaf = (lambda inf, inp : inp)
    result = {v : result[v] if v not in save_leaves else choose_leaf(result[v], save_leaves[v]) for v in result if (v != 'fixed-root') or (not add_diploid_root)}
    assert set(result.keys()) == {clade.name for clade in tree.find_clades()}
    iswgd = {v : False for v in result} if iswgd is None else iswgd

    get_base = (lambda parent, child : ((result[parent][0] if parent in result else 1) * (1 + iswgd[child]), 
                                        (result[parent][1] if parent in result else 1) * (1 + iswgd[child])))
    get_even = (lambda child, base : (
        (0 if base[0] <= result[child][0] else base[0] - result[child][0], 
         0 if base[1] <= result[child][1] else base[1] - result[child][1]),
        (0 if base[0] >= result[child][0] else result[child][0] - base[0], 
         0 if base[1] >= result[child][1] else result[child][1] - base[1]),
        iswgd[child]))
    events = {child : get_even(child, get_base(parent, child)) for parent, child in ((node.name, child.name)
                                                                                     for node in T.find_clades(order='preorder', terminal=False) 
                                                                                     for child in node.clades)}
    return result, cost[T.root.name][input_cns[T.root.name]], events
#if is_all else {v : result[v][0] for v in result}, \
def cut_focal_events_by_neighbours(final, res, total_edges):
    cn_cols = [i for i in final.columns if 'cn' in i.lower()]
    alleles = [i.replace('cn', '') for i in cn_cols]
    neighbour_cns = final[['Solution', 'Node', 'Chr', 'Pos', 'method'] + cn_cols].copy()
    neighbour_cns[['leftsame_' + c for c in cn_cols]] = (neighbour_cns
                                                         .groupby(['Solution', 'Node', 'Chr'])[cn_cols]
                                                         .transform(lambda x: (np.diff(x, prepend=-1) == 0)))
    neighbour_cns[['rightsame_' + c for c in cn_cols]] = (neighbour_cns
                                                          .groupby(['Solution', 'Node', 'Chr'])[cn_cols]
                                                          .transform(lambda x: (np.diff(x, append=-1) == 0)))
    neighbour_cns = neighbour_cns[neighbour_cns['method'] == 'fillgaps'].drop(cn_cols + ['method'], axis=1).copy()
    test = (total_edges[['Solution', 'Parent', 'Child']]
            .merge(neighbour_cns, left_on=['Solution', 'Parent'], right_on=['Solution', 'Node'], how='left')
            .merge(neighbour_cns, left_on=['Solution', 'Child', 'Chr', 'Pos'], 
                   right_on=['Solution', 'Node', 'Chr', 'Pos'], how='left', suffixes=['_P', '_C']))
    for a in alleles:
        test[f'use_events{a}'] = (~((test[f'leftsame_cn{a}_P'] & test[f'leftsame_cn{a}_C']) | 
                                    (test[f'rightsame_cn{a}_P'] & test[f'rightsame_cn{a}_C'])))
    test = test[['Solution', 'Node_C', 'Chr', 'Pos'] + [f'use_events{a}' for a in alleles]].rename(columns={'Node_C' : 'Node'})
    res_test = res.merge(test, on=['Solution', 'Node', 'Chr', 'Pos'], how='left')
    res_test[[f'use_events{a}' for a in alleles]] = (res_test[[f'use_events{a}' for a in alleles]]
                                                     .astype(float)
                                                     .fillna(1)
                                                     .astype(bool))
    for a in alleles:
        res_test[f'Deletions{a}'] = res_test[f'Deletions{a}'] * res_test[f'use_events{a}']
        res_test[f'Gains{a}'] = res_test[f'Gains{a}'] * res_test[f'use_events{a}']
    return res_test

def add_focal_events_to_edgesdf(res, total_edges, interval_focal=False, final=None):
    cn_cols = [i for i in res.columns if 'cn' in i.lower()]
    alleles = [i.replace('cn', '') for i in cn_cols]
    if interval_focal:
        res = cut_focal_events_by_neighbours(final, res, total_edges) 
    focal_events = (res
                    .assign(focal_losses=sum([res[f'Deletions{a}'] for a in alleles]), 
                            focal_gains=sum([res[f'Gains{a}'] for a in alleles]))
                    .groupby(['Solution', 'Node'])[['focal_losses', 'focal_gains']]
                    .sum()
                    .reset_index()
                    .rename(columns={'Node' : 'Child'}))
    focal_events['Child'] = focal_events['Child'].astype(str)
    total_edges = total_edges.merge(focal_events, on=['Solution', 'Child'], how='left')
    total_edges['Deletions'] = total_edges['Deletions'] + total_edges['focal_losses']
    total_edges['Gains'] = total_edges['Gains'] + total_edges['focal_gains']
    total_edges['Events'] = total_edges['Deletions'] + total_edges['Gains']
    focal_label = '_interval' if interval_focal else ''
    total_edges[f'Total_cost_focal{focal_label}'] = total_edges.groupby(['Solution'])['Events'].transform('sum').astype(int)
    return total_edges[f'Total_cost_focal{focal_label}'].copy()

class ProgressBar:

    def __init__(self, total, length, counter=0, verbose=False, decimals=1, fill=chr(9608), prefix = 'Progress:', 
                 suffix = 'Complete'):
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
        v = 0 if float(self.total) == 0 else (self.counter / float(self.total))
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * v)
        filledLength = 0 if self.total == 0 else int(self.length * self.counter // self.total)
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

