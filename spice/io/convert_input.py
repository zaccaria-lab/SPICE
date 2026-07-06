import pandas as pd
import numpy as np
import scipy

def convert_sim_major_minor(df, major_minor_type):
    id_cols = ['Node', 'Chr', 'Pos', 'Start', 'End']
    df_test = df.pivot_table(index=id_cols, columns='Allele', values='CN').reset_index()
    df_test['switch'] = np.where(df_test['A'] == df_test['B'], 'SAME', df_test['A'] > df_test['B'])
    if major_minor_type == 'leaf_dependent':
        switch_pos = (df_test[df_test.switch != 'SAME']
                      .groupby(['Chr', 'Pos', 'Start', 'End'])['switch']
                      .apply(lambda x: x.value_counts().index[0])
                      .reset_index())
        df_test = df_test.drop(['switch'], axis=1).merge(switch_pos, on=['Chr', 'Pos', 'Start', 'End'], how='left')
        df_test['switch'] = df_test['switch'].fillna(False) 
    df_test['minor'] = np.where(df_test['switch'], df_test['B'], df_test['A']) 
    df_test['major'] = np.where(df_test['switch'], df_test['A'], df_test['B']) 
    df_test = df_test[id_cols + ['major', 'minor']]
    df_test.columns = id_cols + ['A', 'B']
    df_test = (pd.melt(df_test, id_vars=['Node', 'Chr', 'Pos', 'Start', 'End'], var_name='Allele', value_name='CN')
               .sort_values(by=id_cols + ['Allele'])
               .reset_index(drop=True))
    return df_test


def sim_to_medicc(nodes, edges=None):
    if edges is not None:
        leaf_nodes = edges[edges.Child_is_leaf]['Child'].values
    else:
        leaf_nodes = nodes.Node.unique()

    leaf_profiles = nodes[nodes.Node.isin(leaf_nodes)].sort_values(by=['Node', 'Chr', 'Allele', 'Pos'])
    leaf_profiles = (leaf_profiles
                     .pivot_table(index=['Node', 'Chr', 'Start', 'End'], columns=['Allele'], values='CN')
                     .reset_index())
    leaf_profiles.columns = ['sample_id', 'chrom', 'start', 'end', 'cn_a', 'cn_b']
    leaf_profiles.sample_id = leaf_profiles.sample_id.apply(lambda x: 'sample_' + str(x))
    return leaf_profiles


def get_breakpoints(leaf_profiles, allele_specific=True):
    leaf_df = pd.DataFrame(leaf_profiles).reset_index()
    leaf_df = pd.melt(leaf_df, id_vars='index')
    leaf_df.columns = ['Chr', 'Node', 'CN']
    if allele_specific:
        leaf_df['Allele'] =  leaf_df.CN.apply(lambda x: ['A', 'B'])
        leaf_df = leaf_df.explode(['CN', 'Allele'])
    leaf_df['Pos'] = leaf_df.CN.apply(lambda x : np.arange(1, len(x)+1))
    leaf_df = leaf_df.explode(['Pos', 'CN'])
    use_cols = ['Chr', 'Allele', 'Pos'] if allele_specific else ['Chr', 'Pos']
    leaf_df = leaf_df.pivot_table(index='Node', columns=use_cols, values='CN')
    a = leaf_df.to_numpy() 
    return sum(1 - (a[0] == a).all(0))


def convert_to_newick(edges, initial_cc=1):
    return '(' + inner_get_newick_tree(edges, initial_cc) + ',0:0):0'

def inner_get_newick_tree(edges, node):
    parent = str(node) + ':' + str([v for k, v in edges.items() if k[1] == node][0])
    children = [':'.join([str(k[1]), str(v)]) for k, v in edges.items() if k[0] == node]
    if len(children) == 0:
        return parent 
    else:
        children = [inner_get_newick_tree(edges, k[1]) for k, v in edges.items() if k[0] == node]
        return '(' + ','.join(children) + ')INT' + parent 