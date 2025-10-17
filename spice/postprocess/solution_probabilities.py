import pandas as pd
import numpy as np
from scipy.stats import geom, pareto

from spice.io.logging import *


def get_solution_probabilities(edges, nodes, logger=None, gap=0, weight_func='geometric', cost_col='Total_cost', max_prop=1.5):
    
    print('weight func = ', weight_func)
    weighted = False if str(weight_func) == 'False' else True
    
    if logger is None:
        logger, logfile = initialize_logger(True, None)
        close = True
    else:
        close = False
        
    if gap > 0:
        logger.warning('ILP gap = {gap} - Reducing solutions for ranking')
        edges = edges[edges.Total_cost <= 1.05 * edges.Total_cost.min()].copy() 
        use_solutions = edges['Solution'].unique()
        nodes = nodes[nodes.Solution.isin(use_solutions)]
    
    logger.trace('Getting unique trees')  
    edges = weight_trees(edges, cost_col, weight_func, max_prop)
    c = 'Tree_weight'
    trees = get_unique_trees(edges, tree_weight_col=c)
    logger.trace('Calculating topology probabilities')  
    tree_paths = get_edge_paths(trees, tree_weight_col=c)
    tree_probabilities = calculate_tree_probabilities(tree_paths, tree_weight_col=c)
    
    top_tree = tree_probabilities[['tree_id', 'topology_prob']].drop_duplicates().sort_values('topology_prob', ascending=False)['tree_id'][0]
    
    logger.trace('Calculating copy-number state probabilities')  
    cn_paths, proc_nodes = get_node_paths(nodes, tree_paths, tree_weight_col=c)
    cn_probabilities = calculate_cn_probabilities(cn_paths, proc_nodes, tree_weight_col=c)

    chosen_top_cn_probabilities = calculate_cn_position_probabilities(cn_paths[cn_paths.tree_id == top_tree], proc_nodes, tree_weight_col=c).assign(tree_id=top_tree)
    
    logger.trace('Calculating tree rankings')  
    total_probabilities = calculate_tree_rankings(tree_probabilities, cn_probabilities)
    total_probabilities = trees.merge(total_probabilities, on=['tree_id', 'Solution']).merge(edges[['Solution', cost_col, 'Gurobi_cost']].drop_duplicates(), on='Solution')
    
    if close:
        logger.removeHandler(logfile)
        logger.handlers.clear()
        
    return total_probabilities, chosen_top_cn_probabilities

get_weights_sum = (lambda x, max_prop: np.sum((x.max() * max_prop) - x + 1))
get_linear_max_weight = (lambda x, max_prop=1.1: dict(zip(x, ((np.max(x) * max_prop) - x + 1)/get_weights_sum(x, max_prop))))
get_exp_min_weight = (lambda x : dict(zip(x, np.exp((x.min() - x))))) #x.min() 
get_exp2_min_weight = (lambda x : dict(zip(x, np.exp2((x.min() - x)))))
get_sigmoid_weight = (lambda x : dict(zip(x, 1/(1 + np.exp(-(x.min() - x))))))
get_geometric_weight = (lambda x: dict(zip(x, geom.pmf(k=[i + 1 - min(x) for i in x], p=0.8)))) # change p to func of min and max e.g. min(x)/max(x)??
get_pareto_weight = (lambda x: dict(zip(x, pareto.pdf(x=[i + 1 - min(x) for i in x], b=1)))) # change p to func of min and max e.g. min(x)/max(x)??

def weight_trees(edges, cost_col='Total_cost', weight_func='linear_max', max_prop=1.5):
    min_cost_use = edges[~edges['Solution'].str.contains('o')][cost_col].min()
    edges['weight_cost_col'] = np.clip(edges[cost_col], a_min=min_cost_use, a_max=None)
    
    if weight_func == 'og':
        weights_sum = np.sum(np.max(edges['weight_cost_col']) - edges[['Solution', 'weight_cost_col']].drop_duplicates()['weight_cost_col'].values + 1)
        edges['Tree_weight'] = (np.max(edges['weight_cost_col']) - edges['weight_cost_col'] + 1)/weights_sum
    else:
        if weight_func == 'linear_max':
            weight_mapping = get_linear_max_weight(edges['weight_cost_col'].unique(), max_prop)
        elif weight_func == 'exp_min':
            weight_mapping = get_exp_min_weight(edges['weight_cost_col'].unique())
        elif weight_func == 'exp2_min':
            weight_mapping = get_exp2_min_weight(edges['weight_cost_col'].unique())
        elif weight_func == 'sigmoid':
            weight_mapping = get_sigmoid_weight(edges['weight_cost_col'].unique())
        elif weight_func == 'geometric':
            weight_mapping = get_geometric_weight(edges['weight_cost_col'].unique())
        elif weight_func == 'pareto':
            weight_mapping = get_pareto_weight(edges['weight_cost_col'].unique())
        elif weight_func == False:
            weight_mapping = dict(zip(edges['weight_cost_col'].unique(), np.repeat(1, len(edges[cost_col].unique()))))
        else:
            raise ValueError('Invalid weighting method')
        edges['Tree_weight'] = edges['weight_cost_col'].map(weight_mapping)
    return edges.drop(['weight_cost_col'], axis=1)


def get_unique_trees(edges, tree_weight_col='Tree_weight'):
    edges['Branch'] = list(zip(edges.Parent.astype(str), edges.Child.astype(str)))
    if 'tree_id' in edges.columns:
        trees = edges.groupby(['tree_id', 'Solution', tree_weight_col])['Branch'].apply(lambda x: tuple(np.sort(x))).reset_index().rename(columns={'Branch' : 'Tree'})
    else:
        trees = edges.groupby(['Solution', tree_weight_col])['Branch'].apply(lambda x: tuple(np.sort(x))).reset_index().rename(columns={'Branch' : 'Tree'})
        unique_trees = {j : i+1 for i, j in enumerate(trees['Tree'].unique())}
        trees['tree_id'] = trees['Tree'].apply(lambda x: unique_trees[x])
    return trees

def get_edge_paths(trees, tree_weight_col='Tree_weight'):
    unique_trees = trees[['Tree', 'tree_id']].drop_duplicates()
    res = pd.concat(list(map(lambda x: get_tree_ancestors(x), unique_trees.values)))
    unique_paths = {j : i + 1 for i, j in enumerate(res.NodePath.unique())}
    res['path_id'] = list(map((lambda x: unique_paths[x]), res['NodePath']))
    trees_res = res.merge(trees[['tree_id', 'Solution', tree_weight_col]], on='tree_id')
    return trees_res

def get_paths(branch, tree, anc_dict):
    if (branch in anc_dict):
        return anc_dict
    else:
        if branch != ('internal_0', 'internal_1'):
            parent_branch = [i for i in tree if i[1] == branch[0]][0]
            ancestors = get_paths(parent_branch, tree, anc_dict)
            path = tuple(sorted(set([branch[0]] + list(ancestors[parent_branch]))))
        else:
            path = tuple([branch[0]])
        anc_dict[branch] = path
        return anc_dict 
    
def get_tree_ancestors(tree):
    tree, tree_id = tree
    ancestors = {}
    for branch in tree:
        ancestors = get_paths(branch, tree, ancestors)
    out = pd.Series(ancestors, name='NodePath').reset_index().rename(columns={'level_0' : 'Parent', 'level_1' : 'Node'}).assign(tree_id=tree_id)
    return out.assign(branch=tuple(zip(out.Parent, out.Node)))


def calculate_tree_probabilities(trees_res, tree_weight_col='Tree_weight'):
    trees_prob = trees_res[['tree_id', 'Solution', tree_weight_col, 'Node', 'branch', 'path_id']].drop_duplicates().copy()
    trees_prob['fract_branch'] = trees_prob.groupby(['path_id', 'Node', 'branch'])[tree_weight_col].transform('sum')
    trees_prob = trees_prob.merge(trees_prob[['path_id', 'Solution', tree_weight_col]].drop_duplicates().groupby('path_id')[tree_weight_col].sum().reset_index().rename(columns={tree_weight_col : 'fract_path'}), on=['path_id'])
    trees_prob['topology_prob'] = np.where(trees_prob['fract_path'] > 0, trees_prob['fract_branch'] / trees_prob['fract_path'], 0)
    solution_probabilities = trees_prob.groupby(['tree_id', 'Solution'])['topology_prob'].apply(lambda x: sum(np.log(list(x))))
    return solution_probabilities.reset_index()


def get_node_paths(nodes, trees_paths, tree_weight_col='Tree_weight'):  
    internal_nodes = trees_paths.Parent.unique()
    nodes = nodes[nodes.Node.isin(internal_nodes)].copy()
    nodes = pd.melt(nodes, id_vars=['Solution', 'Node', 'Chr', 'Pos', 'Start', 'End'], value_vars=['cnA', 'cnB'], var_name='Allele', value_name='cn')
    nodes['position'] = nodes['Chr'].astype(str) + '_' + nodes['Pos'].astype(str) + '_' + nodes['Allele'].str.strip('cn')
    nodes_res = nodes.pivot_table(index=['Solution', 'Node'], columns='position', values='cn').fillna(1).astype(int)
    pos = tuple(nodes_res.columns)
    nodes_res = pd.Series(list(map(tuple, nodes_res.values)), index=nodes_res.index, name='cn').reset_index()
    total_res = nodes_res.rename(columns={'Node' : 'NodePath'}).merge(trees_paths[trees_paths.Node.isin(internal_nodes)].explode('NodePath'), on=['Solution', 'NodePath'], how='inner').sort_values(by=['path_id', 'Solution', 'Node', 'NodePath'])
    cn_paths = total_res.groupby(['Solution', tree_weight_col, 'tree_id', 'path_id', 'Node'])['cn'].apply(lambda x: tuple(map(tuple, np.vstack(x).T))).reset_index() 
    cn_paths['position'] = [pos]*len(cn_paths)
    cn_paths = cn_paths.explode(['cn', 'position']).reset_index(drop=True).rename(columns={'cn' : 'cnPath'})
    cn_path_mapping = dict(zip(np.sort(cn_paths['cnPath'].unique()), range(1, cn_paths['cnPath'].nunique()+1)))
    cn_paths['cn_path_id'] = list(map(lambda x: cn_path_mapping[x], cn_paths['cnPath']))
    cn_paths = cn_paths.merge(trees_paths[['Solution', 'Node', 'NodePath']], on=['Solution', 'Node'], how='left')
    return cn_paths, nodes


def calculate_cn_probabilities(cn_paths, nodes, tree_weight_col='Tree_weight'):
    total_mapping = cn_paths[['Solution', tree_weight_col, 'Node', 'position', 'path_id', 'cn_path_id']].drop_duplicates()
    cn_prob = cn_paths.merge(nodes[['Solution', 'Node', 'position', 'cn']].drop_duplicates().rename(columns={'cn' : 'cn_branch_id'}), on=['Solution', 'Node', 'position'], how='left')
    cn_prob['fract_cn'] = cn_prob.groupby(['path_id', 'Node', 'position', 'cn_path_id', 'cn_branch_id'])[tree_weight_col].transform('sum')
    cn_prob['fract_path'] = cn_prob.groupby(['path_id', 'Node', 'position', 'cn_path_id'])[tree_weight_col].transform('sum')     
    cn_prob['cn_prob'] = np.where(cn_prob['fract_path'] > 0, cn_prob['fract_cn'] / cn_prob['fract_path'], 0)
    assert (cn_prob['cn_prob'] <= 1).all(), 'CN probabilities should be between 0 and 1'
    assert (cn_prob['cn_prob'] >= 0).all(), 'CN probabilities should be between 0 and 1'
    assert (cn_prob['fract_path'] >= 0).all()
    cn_probabilities = cn_prob.groupby(['Solution'])['cn_prob'].apply(lambda x: np.sum(np.log(x))).reset_index()
    return cn_probabilities 


def calculate_cn_position_probabilities(cn_paths, nodes, tree_weight_col='Tree_weight'):
    total_mapping = cn_paths[['Solution', tree_weight_col, 'NodePath', 'Node', 'position', 'path_id', 'cn_path_id']].drop_duplicates()
    pos_mapping = nodes[['position', 'Chr', 'Pos', 'Start', 'End', 'Allele']].drop_duplicates()
    cn_prob = cn_paths.merge(nodes[['Solution', 'Node', 'position', 'cn']].drop_duplicates().rename(columns={'cn' : 'cn_branch_id'}), on=['Solution', 'Node', 'position'], how='left')
    cn_prob['fract_cn'] = cn_prob.groupby(['path_id', 'Node', 'position', 'cn_path_id', 'cn_branch_id'])[tree_weight_col].transform('sum')
    cn_prob['fract_path'] = cn_prob.groupby(['path_id', 'Node', 'position', 'cn_path_id'])[tree_weight_col].transform('sum')     
    cn_prob['cn_prob'] = np.where(cn_prob['fract_path'] > 0, cn_prob['fract_cn'] / cn_prob['fract_path'], 0)
    assert (cn_prob['cn_prob'] <= 1).all(), 'CN probabilities should be between 0 and 1'
    assert (cn_prob['cn_prob'] >= 0).all(), 'CN probabilities should be between 0 and 1'
    assert (cn_prob['fract_path'] >= 0).all()
    cn_prob_mapping = cn_prob.groupby(['Solution', 'position'])[['Node', 'cn_prob']].apply(lambda x: {'internal_0' : 1} | dict(zip(x.Node, x.cn_prob))).rename('cn_prob_mapping').reset_index()
    cn_probabilities = cn_prob.merge(cn_prob_mapping, on=['Solution', 'position'], how='left')
    cn_probabilities['path_prob'] = cn_probabilities.apply((lambda x: np.prod([x['cn_prob_mapping'][node] for node in x['NodePath']])), axis=1)
    cn_probabilities['cn_path_prob'] = np.round(cn_probabilities['cn_prob'] * cn_probabilities['path_prob'], 5)
    cn_probabilities = cn_probabilities[['Node', 'position', 'cn_path_id', 'cn_path_prob', 'cn_branch_id']].drop_duplicates() \
                                                                                                           .groupby(['Node', 'position', 'cn_branch_id'])['cn_path_prob'] \
                                                                                                           .sum() \
                                                                                                           .reset_index() \
                                                                                                           .rename(columns={'cn_branch_id' : 'CN', 
                                                                                                                            'cn_path_prob' : 'CN_probability'})
    cn_probabilities = cn_probabilities.merge(pos_mapping, on='position', how='left')   
    assert np.all(np.round(cn_probabilities.groupby(['Node', 'Chr', 'Pos', 'Allele'])['CN_probability'].sum().unique(), 2) == 1), 'Sum of probabilities for each position should be 1.'                                                                                               
    return cn_probabilities[['Node', 'Chr', 'Pos', 'Start', 'End', 'Allele', 'CN', 'CN_probability']].copy()


def calculate_tree_rankings(tree_probabilities, cn_probabilities):
    total_probabilities = tree_probabilities.merge(cn_probabilities, on='Solution')
    # non-normalised probabilities
    total_probabilities['solution_prob'] = total_probabilities['topology_prob'] + total_probabilities['cn_prob']
    # normalised probabilities
    total_probabilities['topology_prob_norm'] = (np.abs(total_probabilities['topology_prob']) / np.abs(total_probabilities['topology_prob']).max()).fillna(0)
    total_probabilities['cn_prob_norm'] = (np.abs(total_probabilities['cn_prob']) / np.abs(total_probabilities['cn_prob']).max()).fillna(0)
    total_probabilities['solution_prob_norm'] = total_probabilities['topology_prob_norm'] + total_probabilities['cn_prob_norm']
    total_probabilities = total_probabilities.sort_values(by='solution_prob', ascending=False)
    # get rankings - topxcn, topxcn_norm, top>cn
    rankings = {j : i+1 for i, j in enumerate(sorted(total_probabilities['solution_prob'].unique(), reverse=True))}
    rankings_norm = {j : i+1 for i, j in enumerate(sorted(total_probabilities['solution_prob_norm'].unique()))}
    topology_rankings = {j : i+1 for i, j in enumerate(sorted(total_probabilities['topology_prob'].unique(), reverse=True))}
    total_probabilities['topology_rank'] = list(map(lambda x: topology_rankings[x], total_probabilities['topology_prob']))
    topology_effect = total_probabilities.sort_values(by='topology_rank').groupby(['topology_rank'])['Solution'].nunique().shift().fillna(0).cumsum().to_dict()
    topology_cn_prob = total_probabilities.groupby('topology_rank')['cn_prob'].unique().apply(lambda x: sorted(x, reverse=True))
    cn_rankings = {(tree_id, j): int(tree_add + i + 1) for tree_id, tree_add in topology_effect.items() for i, j in enumerate(topology_cn_prob[tree_id])}
    total_probabilities['top>cn_solution_rank'] = list(map(lambda x, y: cn_rankings[(x, y)], total_probabilities['topology_rank'], total_probabilities['cn_prob']))
    total_probabilities['topxcn_solution_rank'] = total_probabilities['solution_prob'].map(rankings) 
    total_probabilities['topxcn_norm_solution_rank'] = total_probabilities['solution_prob_norm'].map(rankings_norm)
    return total_probabilities

