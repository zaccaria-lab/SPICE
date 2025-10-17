import numpy as np
import pandas as pd
# from graphviz import Digraph
import seaborn as sns
import matplotlib.pyplot as plt
import os
import igraph as ig

def plot_tree(total_nodes, total_edges, plot_type='cn', sim_nodes=None, label_edges=True, chrs=range(1, 23), allele_specific=True, real_lengths=True, solution=None, savefig=False, returnfig=True):
        
    sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
    n_nodes = len(total_nodes.Node.unique())
    fig = plt.figure(figsize=(30 + int(n_nodes/2), np.max([8, n_nodes])), constrained_layout=True)
    ratios = [8, 0.5, 8, 8] if allele_specific else [8, 0.5, 16]
    gs = fig.add_gridspec(n_nodes+1, 3 + allele_specific, width_ratios=ratios, height_ratios=[0.5] + [1]*n_nodes)
    tree_fig = fig.add_subplot(gs[:, 0])
    tree_fig.set_title('Tree topology', fontsize=24)
    node_fig = {i : fig.add_subplot(gs[i, 1]) for i in range(n_nodes+1)}
    cna_fig = {i : fig.add_subplot(gs[i, 2]) for i in range(n_nodes+1)}
    if allele_specific:
        cna_fig[0].set_title('Allele A CNs', fontsize=24)
        cnb_fig = {i : fig.add_subplot(gs[i, 3]) for i in range(n_nodes+1)}
        cnb_fig[0].set_title('Allele B CNs', fontsize=24)
    else:
        cna_fig[0].set_title('Total CNs', fontsize=24)
        cnb_fig = None
    
    if (plot_type in ['cn', 'eval']) & ('Solution' in total_edges.columns):
        total_edges = total_edges[total_edges.Solution == solution]
    plot_tree_graph(total_edges, tree_fig, label_edges)
    
    plot_nodes(total_nodes, node_fig)
    
    if plot_type == 'cn':
        plot_cn_profiles(total_nodes, chrs, cna_fig, cnb_fig, solution, allele_specific, real_lengths)
    elif plot_type == 'eval':
        assert sim_nodes is not None, 'sim_nodes must be provided for eval plots'
        if solution is not None:
            total_nodes = total_nodes[total_nodes.Solution == solution]
        use_total_nodes = align_spice_segs_to_sim(sim_nodes, total_nodes)
        if 'CN' not in use_total_nodes.columns:
            use_total_nodes = convert_wide_to_long(use_total_nodes)
        use_nodes = sim_nodes.merge(use_total_nodes, on=['Node', 'Chr', 'Pos', 'Start', 'End', 'Allele'], suffixes=['_sim', '_spice'])
        use_nodes['CN'] = (use_nodes['CN_sim'] != use_nodes['CN_spice']).map({True : 6, False: 1})
        use_nodes = use_nodes.drop(['CN_sim', 'CN_spice'], axis=1)
        plot_cn_profiles(use_nodes, chrs, cna_fig, cnb_fig, solution, allele_specific, real_lengths)
    elif plot_type == 'cn_prob':
        plot_cn_probabilities(total_nodes, cna_fig, cnb_fig)
    elif plot_type == 'event_prob':
        plot_event_probabilities(total_nodes, cna_fig, cnb_fig)
    
    sns.despine(left=False, bottom=False, right=False, top=False)
    # plt.subplots_adjust(wspace=0.05)
    
    if savefig != False:
        plt.savefig(savefig, dpi=300)
    if returnfig == False:
        plt.close()
        
def align_spice_segs_to_sim(sim_nodes, spice_nodes):
    sim_pos = {i: j for i, j in sim_nodes[['Chr', 'Pos', 'Start', 'End']].drop_duplicates().groupby('Chr')}
    spice_pos = {i: j for i, j in spice_nodes[['Chr', 'Pos', 'Start']].drop_duplicates().groupby('Chr')}
    merge_dfs = (lambda sim_df, spice_df: pd.merge_asof(left=sim_df, right=spice_df, by='Chr', on='Start', suffixes=['_sim', ''], direction='backward'))
    pos_mapping = pd.concat([merge_dfs(sim_pos[i], spice_pos[i]) for i in spice_pos.keys()])
    return spice_nodes.drop(['Start', 'End'], axis=1).merge(pos_mapping, on=['Chr', 'Pos']).drop('Pos', axis=1).rename(columns={'Pos_sim' : 'Pos'})
        
def plot_tree_graph(edges, ax=None, label_edges=True):

    if type(edges) != pd.DataFrame:
        edges = pd.DataFrame(edges, columns=['Parent', 'Child'])
    
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()

    edge_df = edges.copy()
    
    nodes = np.unique(edge_df[['Parent', 'Child']].values)
    n_nodes = len(nodes)
    nodes_index = range(n_nodes)
    conversion = dict(zip(nodes, nodes_index))

    edge_df['Parent'] = edge_df['Parent'].apply(lambda x: conversion[x])
    edge_df['Child'] = edge_df['Child'].apply(lambda x: conversion[x])

    edge_list = list(zip(edge_df.Parent, edge_df.Child))
    g = ig.Graph(n_nodes, edge_list)
    g.vs["name"] = nodes

    style = {}
    style["margin"] = 30
    style["vertex_color"] = ['white'] * n_nodes #list(sns.color_palette('tab20', n_nodes))
    style['vertex_frame_color'] = 'black'
    style["vertex_size"] = 45
    style["vertex_label_size"] = 20
    style["vertex_label"] = g.vs["name"]
    style["edge_width"] = 4

    if 'branch_color' not in edge_df.columns:
        edge_df['branch_color'] = 'black'
    style["edge_color"] = edge_df.branch_color.values
    style["edge_background"] = 'white'

    layout = g.layout_reingold_tilford(root=[0])
    ig.plot(g, layout=layout, **style, transparent=True, target=ax)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymax, ymin)
    ax.set_xticks([])
    ax.set_yticks([])
    
    if label_edges:
        edge_df['wgd_str'] = np.where(edge_df['WGD'] > 0, '#WGD: ' + edge_df['WGD'].astype(str) + '\n', '')
        edge_df['gains_str'] = '#Gains: ' + edge_df[[i for i in edge_df.columns if 'Gain' in i]].sum(axis=1).astype(str) + '\n'
        edge_df['losses_str'] = '#Losses: ' + edge_df[[i for i in edge_df.columns if 'Loss' in i]].sum(axis=1).astype(str)
        edge_df['branch_str'] = edge_df['wgd_str'] + edge_df['gains_str'] + edge_df['losses_str']
        node_coords = {vertex : coordinate for vertex, coordinate in enumerate(layout)}
        for i, row in edge_df.iterrows():
            parent = row['Parent']
            child = row['Child']
            x1, y1 = node_coords[parent]
            x2, y2 = node_coords[child]
            just = 'center' if x1 == x2 else 'left' if x1 < x2 else 'right'
            t = ax.text((x1+x2)/2, (y1+y2)/2, row['branch_str'], fontsize=15, ha=just, va='center', color='black')
            t.set_bbox(dict(facecolor='white', alpha=0.8))
    
    
def plot_nodes(df, nodes_ax):
    n_nodes = len(df.Node.unique())
    node_cmap = dict(zip(sorted(df.Node.unique()), sns.color_palette('tab20', n_nodes).as_hex()))
    node_ids = {i : node for i, node in enumerate(sorted(df.Node.unique()))}

    for node_id, node in node_ids.items():
        sns.scatterplot(x=[1], y=[1], s=1200, edgecolor='black', linewidth=2, alpha=0.8, color=node_cmap[node], ax=nodes_ax[node_id+1])
        nodes_ax[node_id+1].text(x=0.4, y=0.5, s=str(node), color='black', fontsize=16)
        nodes_ax[node_id+1].set_frame_on(False)
        nodes_ax[node_id+1].set_xlim(-1, 3)
        nodes_ax[node_id+1].set_ylim(-1, 3)
        nodes_ax[node_id+1].set_xticks([])
        nodes_ax[node_id+1].set_yticks([])
        
    nodes_ax[0].set_ylabel('')
    nodes_ax[0].set_xticks([])
    nodes_ax[0].set_yticks([]) 
    
    
def plot_cn_profiles(node_df, chrs, a_ax, b_ax, solution=None, allele_specific=True, real_lengths=True):
    if allele_specific:
        if 'CN' not in node_df.columns:
            node_df = convert_wide_to_long(node_df)
    else:
        if 'CN' not in node_df.columns:
            node_df['CN'] = node_df['cnA'] + node_df['cnB']
        else:
            columns = [i for i in node_df.columns if i not in ['Allele', 'CN', 'Positions']]
            node_df = node_df.groupby(columns)['CN'].sum().reset_index()
        node_df['Allele'] = 'A'
    if solution is not None:
        node_df = node_df[node_df.Solution == solution].copy()

    node_df.columns = node_df.columns.get_level_values(0)

    df = node_df[node_df.Chr.isin(chrs)].copy().reset_index(drop=True)
    df['Allele'] = df['Allele'].map({'A' : 0, 'B' : 1})
    if real_lengths:
        df['Positions'] = df.apply((lambda x: range(x.Start, x.End, 1000000)), axis=1)
        df = df.explode('Positions').reset_index(drop=True)
    else:
        df['Positions'] = df['Pos']
    n_nodes = len(df.Node.unique())
    node_cmap = dict(zip(sorted(df.Node.unique()), sns.color_palette('tab20', n_nodes).as_hex())) #
    node_ids = {i : node for i, node in enumerate(sorted(df.Node.unique()))} #
    
    for j, allele in enumerate([0, 1]):
        use_ax = a_ax if allele == 0 else b_ax
        chr_data = df[(df.Node == node_ids[0])]
        chrom = chr_data[(chr_data.Allele == allele)][['Chr','Positions']].drop_duplicates().sort_values(by=['Chr', 'Positions']).reset_index(drop=True)
        chrom['chr_code'] = ((chrom['Chr']-1) % 3)
        chr_data = chrom['chr_code'].to_frame().T  
        chr_labels = chrom.Chr.unique()
        chr_idxs = list(chrom[chrom.Chr != chrom.Chr.shift()].index) + [chrom.index[-1]]
        chr_idxs = {chr_labels[i]: chr_idxs[i] + ((chr_idxs[i+1] - chr_idxs[i])/2) for i in range(len(chr_idxs)-1)}
        sns.heatmap(chr_data, cmap='Greys', vmax=5, vmin=-1, ax=use_ax[0], cbar=False)
        for c, pos in chr_idxs.items():
            y = 0.6 if (c % 2) == 0 else 0.9
            use_ax[0].text(x=pos, y=y, s=c, color='black', fontsize=16)
        use_ax[0].set_ylabel('')
        use_ax[0].set_xticks([])
        use_ax[0].set_yticks([])
    
    for node_id, node in node_ids.items():
        total_data = df[(df.Node == node)].copy().reset_index(drop=True)

        for j, allele in enumerate([0, 1]):
            use_ax = a_ax if allele == 0 else b_ax
            data = total_data[total_data.Allele == allele][['Chr', 'CN', 'Positions']].drop_duplicates().copy().reset_index(drop=True)
            plot_data = data['CN'].to_frame().T
            sns.heatmap(plot_data, cmap='coolwarm', vmax=6, center=1, vmin=0, ax=use_ax[node_id+1], cbar=False) 
            use_ax[node_id+1].vlines(x=data[data.Chr != data.Chr.shift()].index, ymin=0, ymax=1, color='black', linestyles='dashed')
            use_ax[node_id+1].set_ylabel(node, rotation=0)
            use_ax[node_id+1].set_xticks([])
            use_ax[node_id+1].set_yticks([])

    plt.rcParams['figure.edgecolor'] = 'white'

def convert_wide_to_long(total_nodes):
    id_vars = [i for i in total_nodes.columns if 'cn' not in i]
    cn = pd.melt(total_nodes, id_vars=id_vars, value_vars=['cnA', 'cnB'], var_name='Allele', value_name='CN')
    cn['Allele'] = cn['Allele'].map({'cnA' : 'A', 'cnB' : 'B'})
    return cn

def convert_long_to_wide(total_nodes):
    total_nodes['Allele'] = total_nodes['Allele'].map({0 : 'cnA', 1 : 'cnB'})
    cn = total_nodes.pivot_table(index=['Solution', 'Node', 'Chr', 'Pos', 'Start', 'End', 'method'], columns=['Allele'], values='CN').reset_index()
    return cn


# plot cn probabilities

def plot_cn_probabilities(node_df, a_ax, b_ax):
    
    if 'cnA' not in node_df.columns:
        node_df = convert_long_to_wide(node_df)
    
    n_nodes = node_df.Node.nunique()
    max_cn = node_df[['cnA', 'cnB']].max().max()
    
    v_list_a = get_plot_data(node_df, 'cnA')
    v_list_b = get_plot_data(node_df, 'cnB')
    for node in range(n_nodes):
        for d, ax in zip(['cnA', 'cnB'], [a_ax, b_ax]):
            use_list = v_list_a if d == 'cnA' else v_list_b
            data = use_list[node]
            sns.heatmap(process_cn_data(data, d, max_cn), cmap='crest', vmin=0, vmax=1, ax=ax[node], cbar=False)
            chr_data = data.T.reset_index()[2:]
            ax[node].vlines(x=chr_data[chr_data.Chr != chr_data.Chr.shift()].index, ymin=0, ymax=max_cn, color='black', linestyles='dashed')
            ax[node].set_ylabel('')
            ax[node].set_xticks([])
            
def process_cn_data(g, cn, max_cn):
    node = g.Node.values[0]
    if node == 0:
        g = g.replace({0 : 1})
    g = g.drop(['Node'], axis=1, level=0).set_index(cn).T
    need_cols = range(max_cn+1)
    cols = list(g.columns)
    for i in need_cols:
        if i not in cols:
            g[i] = 0
    return g.T.sort_index(ascending=False).replace({0: np.nan})

def get_plot_data(spice, cn):
    print(f'getting plot data: {cn}')
    v = spice.pivot_table(index=['Node', cn], columns=['Chr', 'Pos'], values='Solution', aggfunc='count')
    v = v.fillna(0) / spice.Solution.nunique()
    v.reset_index(inplace=True)
    v = v.sort_values(by=['Node', cn], ascending=[True, False])
    v_list = [q for p, q in v.groupby(['Node'])]
    return v_list 
        

# plot event probabilities:

def plot_event_probabilities(node_df, a_ax, b_ax):
    
    if 'cnA' not in node_df.columns:
        node_df = convert_long_to_wide(node_df)
    
    n_nodes = node_df.Node.nunique()
    v_list_a = (get_event_freq_data(node_df, 'GainsA'), get_event_freq_data(node_df, 'LossesA'))
    v_list_b = (get_event_freq_data(node_df, 'GainsB'), get_event_freq_data(node_df, 'LossesB'))
    for node in range(n_nodes):
        for use_list, ax in zip([v_list_a, v_list_b], [a_ax, b_ax]):
            sns.lineplot(x='plotPos', y='Probability', data=process_data(use_list[0][node]), ax=ax[i, 0], color='#C0392B', linewidth=4, alpha=0.7, drawstyle='steps-pre')
            sns.lineplot(x='plotPos', y='Probability', data=process_data(use_list[1][node]), ax=ax[i, 0], color='#2471A3', linewidth=4, alpha=0.7, drawstyle='steps-pre')
            
            chr_data = use_list[0][node].T.reset_index()[2:]
            ax[node].vlines(x=chr_data[chr_data.Chr != chr_data.Chr.shift()].index, ymin=0, ymax=max_cn, color='black', linestyles='dashed')
            ax[node].set_ylabel('')
            ax[node].set_xticks([])
        # if i != 5:
        #     ax[i, 0].set_xlabel('')
        #     ax[i, 1].set_xlabel('')
        # ax[i, 0].tick_params(left=False, bottom=False, rotation=90)
        # ax[i, 1].tick_params(left=False, bottom=False, rotation=90)
        # ax[0, 0].set_title('n_trees : {}'.format(spice.Solution.nunique()))
        # sns.despine(left=False, top=False, right=False, ax=ax[i, 0])
        # sns.despine(left=False, top=False, right=False, ax=ax[i, 1])
    
def get_event_freq_data(spice, event):
    print(f'getting plot data: {event}')
    spice[event + '_pres'] = (spice[event] >= 1) 
    v = spice.pivot_table(index=['Node'], columns=['Chr', 'Pos'], values=event + '_pres', aggfunc='sum').fillna(0) / spice.Solution.nunique()
    v = v.sort_values(by=['Node'], ascending=[True])
    v_list = [q for p, q in v.groupby(['Node'])]
    return v_list

def process_data(v):
    v = v.reset_index(drop=True).T.reset_index().rename(columns={0 : 'Probability'})
    v['plotPos'] = v['Chr'].astype(str) + '_' + v['Pos'].astype(str)
    return v

