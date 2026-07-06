import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import transforms
from matplotlib import colors as mcolors
import matplotlib.cm

p = '#E8E79AFF, #C2D89AFF, #8CBF9AFF, #5FA2A4FF, #477B95FF, #315B88FF, #24396BFF, #191F40FF'.split(', ') # blues
# p = ['#fed0bb', '#fcb9b2', '#BD665A', '#8C2F30', '#641220', '#430812', '#180105', 'black'] # pink - black
event_prob_cmap = mcolors.LinearSegmentedColormap.from_list('event_prob_cmap', p, N=256) 

def cmap_exists(name):
    try:
        matplotlib.cm.get_cmap(name)
        return True
    except ValueError:
         pass
    return False

if not cmap_exists("event_prob_cmap"):
    matplotlib.cm.register_cmap("event_prob_cmap", event_prob_cmap)


def plot_cn_probabilities(cn_out_tree, cn_profiles, cn_probabilities, leaf_order=None, color_mapping={}, ax=None, chrs='all', real_chr_lengths=True, plot_node_names=True, scalex=1, scaley=1):
    tree_id = cn_probabilities['tree_id'].unique()[0]
    cn_profiles = cn_profiles[cn_profiles['tree_id'] == tree_id].copy()
    cn_out_tree = cn_out_tree[cn_out_tree['tree_id'] == tree_id][['Parent', 'Child', 'WGD']].drop_duplicates().copy()
    cn_probabilities, cn_out_tree = process_input_data_format(cn_probabilities, cn_out_tree, chrs)
    parents = cn_out_tree['Parent'].unique()
    children = cn_out_tree['Child'].unique()
    root = str(get_root(parents, children))
    leaves = get_leaves(parents, children)
    cn_all_profiles = concat_cn_dfs(cn_profiles, cn_probabilities, leaves)
    cn_values = np.arange(0, cn_all_profiles['CN'].max() + 1).astype(int)

    # Add figure
    allele_specific = True if cn_all_profiles.Allele.nunique() == 2 else False
    n_nodes = cn_all_profiles.Node.nunique()
    sizey = ((int(n_nodes * (cn_all_profiles['CN'].max()/10)) + 1) * 1.5) * scaley
    sizex =  (25+(5*allele_specific)) * scalex
    gap = 0.8/scalex if plot_node_names else -1
    width_ratios = [5, gap, 15, 15] if allele_specific else [6, gap, 15]
    fig = plt.figure(figsize=(sizex, sizey))
    gs = fig.add_gridspec(n_nodes + 1, 3 + allele_specific, width_ratios=width_ratios, height_ratios=[0.3] + [1] * n_nodes)
    

    # Bottom left - Tree
    tree_ax = fig.add_subplot(gs[1:, 0])
    coords, edges_plot, y_vals, cn_index = get_coord_df(cn_out_tree, root, leaves, leaf_order, color_mapping, False)
    palette = dict(zip(coords['index'], coords['color']))
    add_cn_tree(tree_ax, edges_plot, coords, palette, root, leaves, False, plot_node_names)
    add_wgd(tree_ax, edges_plot, cn_out_tree)
    
    # Top left = colorbar
    # add_cn_legend(tree_ax, palette)
    
    # Bottom right - CNs
    cn_ax = {(allele, node) : fig.add_subplot(gs[1+node_idx, 2 + allele_idx]) for allele_idx, allele in enumerate(sorted(cn_all_profiles.Allele.unique()))
                                                                     for node_idx, node in enumerate(cn_index[::-1])}
    cn_plot_data = {i: process_cn_probabilities_data(j, real_chr_lengths, cn_values[::-1]) for i, j in cn_all_profiles.groupby(['Allele', 'Node'])}
    add_cn_profiles(cn_ax, cn_plot_data, cn_values)
    
    
    
    # Top right = Chrosomes
    chr_data = {allele : cn_plot_data[(allele, children[0])].columns.T.to_frame().reset_index(drop=True) for allele in sorted(cn_all_profiles.Allele.unique())}
    chr_ax = {allele : fig.add_subplot(gs[0, 2 + allele_idx]) for allele_idx, allele in enumerate(sorted(cn_all_profiles.Allele.unique()))}         
    for allele, ax in chr_ax.items():
        title = 'Allele ' + allele[-1] if allele_specific else None
        add_chr_bar(ax, chr_data[allele], title)
        
    plt.subplots_adjust(wspace=0.11, hspace=0.07);
    
    
def process_input_data_format(cn_profiles, cn_out_tree, chrs):
    cn_profiles['Node'] = cn_profiles['Node'].astype(str)
    if chrs != 'all':
        cn_profiles = cn_profiles[cn_profiles['Chr'].isin(chrs)].copy()
    cn_out_tree[['Parent', 'Child']] = cn_out_tree[['Parent', 'Child']].astype(str)
    return cn_profiles, cn_out_tree

get_root = (lambda parents, children: parents[~np.isin(parents, children)][0])
get_leaves = (lambda parents, children: children[~np.isin(children, parents)])

def concat_cn_dfs(cn_profiles, cn_probabilities, leaves):
    allele_specific = True if cn_probabilities.Allele.nunique() == 2 else False
    if allele_specific:
        cn_leaf_profiles = cn_profiles[(cn_profiles['Node'].isin(leaves))][['tree_id', 'Node', 'Chr', 'Pos', 'Start', 'End', 'cnA', 'cnB']].drop_duplicates().copy()
        cn_leaf_profiles = pd.melt(cn_leaf_profiles, id_vars=['tree_id', 'Node', 'Chr', 'Pos', 'Start', 'End'], value_vars=['cnA', 'cnB'], var_name='Allele', value_name='CN')
    else:
        cn_leaf_profiles = cn_profiles[(cn_profiles['Node'].isin(leaves))].copy()
    cn_leaf_profiles['CN_probability'] = 1
    cn_all_profiles = pd.concat([cn_probabilities, cn_leaf_profiles], axis=0).reset_index(drop=True)
    return cn_all_profiles



# CN tree plotting
    
def add_cn_tree(ax, edges_plot, coords, palette, root, leaves, plot_root=False, plot_node_names=True):
    for i, row in edges_plot.iterrows():
        ax.plot([row.x_Parent, row.x_Child], [row.y_Parent, row.y_Child], color='black', linewidth=5, zorder=0)
    for i, row in coords.iterrows():
        if (plot_root == True) or (row['index'] != root):
            ax.plot([row.x, coords['x'].max()+1.5], [row.y, row.y], color='grey', linestyle='-', linewidth=8, zorder=0, alpha=0.15, clip_on=False)
    sns.scatterplot(x='x', y='y', data=coords, legend=False, s=1000, edgecolor='black', linewidth=2, hue=coords['index'], palette=palette, zorder=3, ax=ax, alpha=1)
    ax.yaxis.tick_right()
    if plot_node_names:
        ax.set_yticks(ticks=coords['y'], labels=[i.replace('sample_', '') if i in leaves else f"int_{i}" if ('internal' not in i) & (i != root) else i.replace('internal_', 'int').replace('internal', 'int') for i in coords['name']], fontsize=16, ha='left')
    else:
        ax.set_yticks([])
    ax.tick_params(axis='y', which='major', pad=5, length=0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_xlim(coords['x'].min()-1, coords['x'].max()+1)
    ax.set_ylim(coords['y'].min()-.5, coords['y'].max()+.5)
    sns.despine(left=True, bottom=True, ax=ax)
    ax.spines['right'].set_color('white')
    
def add_wgd(ax, coords, cn_out_tree):
    wgd_col = [i for i in cn_out_tree.columns if 'wgd' in i.lower()][0]
    wgd_branches = cn_out_tree[cn_out_tree[wgd_col] == 1]['Child'].unique()
    wgd_coords = coords[coords['Child'].isin(wgd_branches)].copy()
    wgd_coords['x'] = (wgd_coords['x_Parent'] + wgd_coords['x_Child']) / 2
    wgd_coords['y'] = (wgd_coords['y_Parent'] + wgd_coords['y_Child']) / 2
    sns.scatterplot(x='x', y='y', data=wgd_coords, legend=False, s=350, marker='*', edgecolor='black', linewidth=1, color='gold', zorder=2, ax=ax, alpha=1)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    

# Legend plotting
 
def add_cn_legend(ax, cn_palette):
    ax_cbar = ax.inset_axes(bounds=(0, 0.8, 0.1, 0.3))
    sns.heatmap(pd.DataFrame(np.arange(0, 7)), ax=ax_cbar, cmap=cn_palette, cbar=False, square=True, linewidths=2, linecolor='white', vmin=0, vmax=6, center=1)
    ax_cbar.set_yticks(np.arange(0.5, 7.5), np.arange(0, 7), fontsize=12, rotation=0)
    ax_cbar.invert_yaxis()
    ax_cbar.set_xticks([])
    ax_cbar.set_title('# Copies', fontsize=18, pad=10)
    
    
# CN plotting

def process_cn_probabilities_data(cn_profiles, real_chr_lengths, cn_values):
    if real_chr_lengths:
        cn_profiles['Positions'] = cn_profiles.apply((lambda x: range(x.Start, x.End, 1000000)), axis=1)
        cn_plot_profiles = cn_profiles.explode('Positions').reset_index(drop=True)
    else:
        cn_plot_profiles = cn_profiles.copy()
        cn_plot_profiles['Positions'] = cn_plot_profiles['Start']
    cn_plot = cn_plot_profiles.pivot_table(columns=['Chr', 'Positions'], index='CN', values='CN_probability').reindex(cn_values).fillna(0)
    return cn_plot

def add_cn_profiles(cn_ax, cn_plot_data, cn_values):
    for data_idx, ax in cn_ax.items():
        sns.heatmap(cn_plot_data[data_idx].replace(0, np.nan), cmap=event_prob_cmap, vmax=1, vmin=0, ax=ax, cbar=False, linewidth=0, linecolor='white')
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylabel('') 
        ax.set_yticks(ticks=[i+0.5 for i in list(cn_values)], labels=list(cn_values[::-1]), rotation=0, fontsize=10)
        sns.despine(left=False, bottom=False, ax=ax)
    
    
# Chromosome plotting
    
def add_chr_bar(ax_chr, chr_data, title=None):
    chr_data['chr_code'] = ((chr_data['Chr']-1) % 2)
    chr_data_plot = chr_data['chr_code'].to_frame().T  
    chr_labels = chr_data.Chr.unique()
    chr_idxs = list(chr_data[chr_data.Chr != chr_data.Chr.shift()].index) + [chr_data.index[-1]]
    div = (lambda x: 3 if int(x) < 8 else 4 if int(x) < 18 else 100)
    chr_idxs = {chr_labels[i]: chr_idxs[i] + ((chr_idxs[i+1] - chr_idxs[i])/div(chr_labels[i])) for i in range(len(chr_idxs)-1)} #
    sns.heatmap(chr_data_plot, cmap='Greys_r', vmax=2, vmin=-1, ax=ax_chr, cbar=False, alpha=0.9)
    for c, pos in chr_idxs.items():
        pre = 'chr' if int(c) < 17 else ''
        y = 0.4 if int(c) in [18, 19, 20, 21] else 0.9
        ax_chr.text(x=pos, y=y, s=f'{pre}{c}', color='black', fontsize=12, rotation=30)
    ax_chr.set_ylabel('')
    ax_chr.set_xlabel('')
    ax_chr.set_xticks([])
    ax_chr.set_yticks([])
    
    if title is not None:
        ax_chr.set_title(title, fontsize=20, pad=20)
        

# Functions for coordinates
    
def get_coord_df(cn_out_tree, root, leaves, leaf_order=None, color_mapping={}, plot_root=False):
    x_vals = get_x_order(cn_out_tree, root, plot_root, leaves)
    y_vals = get_y_order(cn_out_tree, root, leaves, leaf_order, plot_root)
    coords = pd.DataFrame({'x': x_vals, 'y' : y_vals}).reset_index().sort_values(by='y').reset_index()
    coords['color'] = coords['index'].apply((lambda x: color_mapping[x] if x in color_mapping else '#FFFFFF'))
    coords['name'] = np.where(coords['index'].astype(str).str.contains('_'), coords['index'].astype(str).str.split('.').str[0], coords['index'].astype(str))
    edges_plot = cn_out_tree[['Parent', 'Child']].astype(str).merge(coords, left_on='Parent', right_on='index').merge(coords, left_on='Child', right_on='index', suffixes=['_Parent', '_Child'])
    coords['y'] = coords['y'] + 0.5
    edges_plot['y_Parent'] = edges_plot['y_Parent'] + 0.5
    edges_plot['y_Child'] = edges_plot['y_Child'] + 0.5
    cn_index = list(y_vals.keys()) if plot_root else [i for i in y_vals.keys() if i != root]
    return coords, edges_plot, y_vals, cn_index


# Functions for x coords

# all leaves on same level
def get_x_order(cn_out_tree, root, plot_root, leaves):
    nodes = np.unique(cn_out_tree[['Parent', 'Child']].values.flatten())
    tree_edges = list(zip(cn_out_tree['Parent'], cn_out_tree['Child']))
    leaf_paths = {leaf: get_leaf_path(leaf, tree_edges, []) for leaf in leaves}
    max_leaf_path = len(max(leaf_paths.values(), key=len))
    x_vals = {str(node): -calculate_fract_path(node, max(get_leaf_descendants(node, tree_edges, leaves, []), key=(lambda c: len(leaf_paths[c])), default=node), leaf_paths)*max_leaf_path for node in nodes}
    if plot_root == False:
        x_vals[str(root)] = x_vals[str(root)]-2
    return x_vals

def get_leaf_path(leaf, tree_edges, path=[]):
    path.append(leaf)
    parent = [i[0] for i in tree_edges if i[1] == leaf]
    if len(parent) == 0:
        return path
    else:
        path = get_leaf_path(parent[0], tree_edges, path)
        return path
    
def get_leaf_descendants(node, tree_edges, leaves, children=[]):
    node_childs = [i[1] for i in tree_edges if i[0] == node]
    if len(node_childs) == 0:
        return children
    else:
        for child in node_childs:
            if child in leaves:
                children.append(child)
            children = get_leaf_descendants(child, tree_edges, leaves, children)
        return children
    
calculate_fract_path = (lambda node, leaf, leaf_paths: 0 if leaf == node else len(leaf_paths[leaf][:leaf_paths[leaf].index(node)+1])/len(leaf_paths[leaf]))


# Functions for y coords

# all nodes have different y coord
def get_y_order(cn_out_tree, root, leaves, leaf_order=None, plot_root=False):
    if leaf_order is None:
        leaf_order = sorted(leaves, key=(lambda x: x if type(x) != str else int(''.join([i for i in x if i.isdigit()]))))
    leaf_order = {l: n for n, l in enumerate(leaf_order)}
    tree_edges = list(zip(cn_out_tree['Parent'], cn_out_tree['Child']))
    initial_cc = [i[1] for i in tree_edges if str(i[0]) == str(root)][0]
    y_order = get_splits((initial_cc, len(get_children(initial_cc, tree_edges, []))), tree_edges, leaf_order, leaves)
    if plot_root:
        y_order.insert(y_order.index(initial_cc)+1, str(root))
        y_order = {str(node): i for i, node in enumerate(y_order)}
    else:
        y_order = {str(node): i for i, node in enumerate(y_order)}
        y_order[str(root)] = y_order[str(initial_cc)]
    return y_order

def get_splits(node, tree_edges, leaf_order, leaves):
    if node[1] == 0:
        return [node[0]]
    else:
        splits = sorted([(child, len(get_children(child, tree_edges, []))) for child in get_node_children(node[0], tree_edges)], key=(lambda x: get_split_order(x[0], tree_edges, leaf_order, leaves)))
        left_split = process_split(splits[:len(splits)//2], tree_edges, leaf_order, leaves)
        right_split = process_split(splits[len(splits)//2:], tree_edges, leaf_order, leaves) 
    return left_split + [node[0]] + right_split

def process_split(split, tree_edges, leaf_order, leaves):
    split_p = []
    for x in split:
        split_p.extend(get_splits(x, tree_edges, leaf_order, leaves))
    return split_p

def get_split_order(node, tree_edges, leaf_order, leaves):
    leaves = get_leaf_descendants(node, tree_edges, leaves, [])
    if len(leaves) == 0:
        return leaf_order[node]
    else:
        return sum(leaf_order[leaf] for leaf in leaves)/len(leaves)
    

def get_children(node, tree_edges, children=[]):
    node_childs = [i[1] for i in tree_edges if i[0] == node]
    if len(node_childs) == 0:
        return children
    else:
        for child in node_childs:
            children.append(child)
            children = get_children(child, tree_edges, children)
        return children
    
get_node_children = (lambda node, tree_edges: [i[1] for i in tree_edges if i[0] == node])