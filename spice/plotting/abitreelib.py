import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import transforms

# Tree plotting

def plot_cn_figure(cn_out_tree, cn_profiles, root, leaf_order=None, cn_palette='RdBu_r', color_mapping={}, ax=None, plot_root=True, idx=None, chrs='all', real_chr_lengths=True, max_cn=None):
    if idx is not None:
        cn_profiles = cn_profiles[cn_profiles['Solution'] == idx].copy()
        cn_out_tree = cn_out_tree[cn_out_tree['Solution'] == idx].copy()
   
    cn_profiles = process_cn_profiles(cn_profiles, chrs)
    if max_cn is None:
        max_cn = cn_profiles[[i for i in cn_profiles.columns if 'cn' in i.lower()]].values.flatten().max()
    
    cn_out_tree[['Parent', 'Child']] = cn_out_tree[['Parent', 'Child']].astype(str)
    root = str(root)
     
    parents = cn_out_tree['Parent'].unique()
    children = cn_out_tree['Child'].unique()
    leaves = children[~np.isin(children, parents)]
    
    coords, edges_plot, y_vals = get_coord_df(cn_out_tree, root, leaves, leaf_order, color_mapping, plot_root)
    
    palette = dict(zip(coords['index'], coords['color']))
    cn_plot, chr_data = process_cn_profiles_data(cn_profiles, root, plot_root, real_chr_lengths)
    
    if plot_root:
        cn_index = list(y_vals.keys())
    else:
        cn_index = [i for i in y_vals.keys() if i != root]
    cn_plot = [cn_plot_df.reindex(cn_index) for cn_plot_df in cn_plot]
    
    allele_specific = True if len(cn_plot) > 1 else False

    sizey = (int(cn_profiles.Node.nunique()*0.4) + 1) #* 1.5 
    if ax is None:
        if allele_specific:
            fig, ax = plt.subplots(1, 4, figsize=(30, sizey), width_ratios=[5, 0.2, 15, 15], sharey='row')
        else:
            fig, ax = plt.subplots(1, 3, figsize=(25, sizey), width_ratios=[6, 0.2, 15], sharey='row')
        
    ax[1].set_visible(False)
    
    # # Top left = colorbar
    add_cn_legend(ax[0], cn_palette, allele_specific, max_cn)
    
    # Top right - chrs
    name = 'Allele A' if allele_specific else 'Total Copy Number'
    add_chr_bar(ax[2], chr_data, name)
    if allele_specific:
        add_chr_bar(ax[3], chr_data, 'Allele B')

    # Bottom left - Tree
    add_cn_tree(ax[0], edges_plot, coords, palette, root, leaves, plot_root)
    add_wgd(ax[0], edges_plot, cn_out_tree)

    # Bottom right - CNs
    for i, cn_profiles in enumerate(cn_plot):
        add_cn_profiles(ax[i+2], cn_profiles, leaves, root, chr_data, coords, cn_palette, allele_specific, max_cn)
    
    if idx is not None: 
        ax[0].text(-5, 0, f'Solution #{idx}', fontsize=18, ha='center', va='center', zorder=10)
        
    plt.subplots_adjust(wspace=0.04, hspace=-0.05);
    

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
    
def add_cn_legend(ax, cn_palette, allele_specific, max_cn):
    center = 1 if allele_specific else 2 if max_cn > 1 else 0.5
    step = 1 if max_cn > 1 else 0.1
    ax_cbar = ax.inset_axes(bounds=(0, 0, 0.1, 0.05*max_cn)) 
    sns.heatmap(pd.DataFrame(np.arange(0, max_cn+1, step)), ax=ax_cbar, cmap=cn_palette, cbar=False, square=True, linewidths=2, linecolor='white', vmin=0, vmax=max_cn, center=center)
    # ax_cbar.set_yticks(np.arange(0.5, max_cn+1.5, step), [f'{i:.1f}' for i in np.arange(0, max_cn+1, step)], fontsize=12, rotation=0)
    ax_cbar.invert_yaxis()
    ax_cbar.set_xticks([])
    ax_cbar.set_title('# Copies', fontsize=18, pad=10)
    
def add_chr_bar(ax, chr_data, title=None):
    ax_chr = ax.inset_axes(bounds=(0, 1, 1, 0.045))#2)
    chr_data['chr_code'] = ((chr_data['Chr']-1) % 2)
    chr_data_plot = chr_data['chr_code'].to_frame().T  
    chr_labels = chr_data.Chr.unique()
    chr_idxs = list(chr_data[chr_data.Chr != chr_data.Chr.shift()].index) + [chr_data.index[-1]]
    div = (lambda x: 3 if int(x) < 8 else 4 if int(x) < 18 else 100)
    chr_idxs = {chr_labels[i]: chr_idxs[i] + ((chr_idxs[i+1] - chr_idxs[i])/div(chr_labels[i])) for i in range(len(chr_idxs)-1)} #
    sns.heatmap(chr_data_plot, cmap='Greys', vmax=3, vmin=-1, ax=ax_chr, cbar=False, alpha=0.5)
    for c, pos in chr_idxs.items():
        pre = 'chr' if int(c) < 17 else ''
        y = 0.4 if int(c) in [18, 19, 20, 21] else 0.9
        ax_chr.text(x=pos, y=y, s=f'{pre}{c}', color='black', fontsize=12, rotation=30)
    ax_chr.set_ylabel('')
    ax_chr.set_xlabel('')
    ax_chr.set_xticks([])
    ax_chr.set_yticks([])
    
    if title is not None:
        ax_chr.set_title(title, fontsize=26, pad=20)
    
def add_cn_tree(ax, edges_plot, coords, palette, root, leaves, plot_root=False):
    sns.scatterplot(x='x', y='y', data=coords, legend=False, s=500, edgecolor='black', linewidth=1, hue=coords['index'], palette=palette, zorder=1, ax=ax, alpha=1)
    for i, row in edges_plot.iterrows():
        ax.plot([row.x_Parent, row.x_Child], [row.y_Parent, row.y_Child], color='black', linewidth=5, zorder=0)
    for i, row in coords.iterrows():
        if (plot_root == True) or (row['index'] != root):
            ax.plot([row.x, coords['x'].max()+1.5], [row.y, row.y], color='grey', linestyle='-', linewidth=8, zorder=0, alpha=0.15, clip_on=False)
    ax.yaxis.tick_right()
    ax.set_yticks(ticks=coords['y'], labels=[i.replace('sample_', '').replace('clone_', '') if i in leaves else f"int_{i}" if ('internal' not in i) & (i != root) else i.replace('internal_', 'int').replace('internal', 'int') for i in coords['name']], fontsize=16, ha='left')
    ax.tick_params(axis='y', which='major', pad=5, length=0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_xlim(coords['x'].min()-1, coords['x'].max()+.5)
    ax.set_ylim(-1, len(coords)+1)
    sns.despine(left=True, bottom=True, ax=ax)
    ax.spines['right'].set_color('white')
    
def add_cn_profiles(ax, cn_plot, leaves, root, chr_data, coords, cn_palette, allele_specific, max_cn):
    ax.set_facecolor('white')
    center = 1 if allele_specific else 2 if max_cn > 1 else 0.5
    sns.heatmap(cn_plot, cmap=cn_palette, vmax=max_cn, center=center, vmin=0, ax=ax, cbar=False, linewidth=0, linecolor='white') #! CHANGE BACK TO 0
    ax.set_ylim(-.5, len(coords)+.5)
    ax.set_xticks([])
    ax.set_yticklabels([i.replace('sample_', '').replace('clone_', '') if i in leaves else f"int_{i}" if ('internal' not in i) & (i != root) else i.replace('internal_', 'int').replace('internal', 'int') for i in list(cn_plot.index)])
    ax.set_xlabel('')
    ax.set_ylabel('')
    c = 'white' if max_cn == 1 else 'black'
    for i in range(len(coords)):
        ax.axhline(y=i, color='white', linewidth=3)  
    for i in list(chr_data[chr_data.Chr != chr_data.Chr.shift()].index):
        ax.axvline(x=i, ymin=0, ymax=1, color=c, linestyle='dotted', linewidth=2)

def process_cn_profiles_data(cn_profiles, root, plot_root, real_chr_lengths):
    if plot_root == False:
        cn_profiles = cn_profiles[cn_profiles.Node != root].copy()
    if real_chr_lengths:
        cn_profiles['Positions'] = cn_profiles.apply((lambda x: range(x.Start, x.End, 1000000)), axis=1)
        cn_plot_profiles = cn_profiles.explode('Positions').reset_index(drop=True)
    else:
        cn_plot_profiles = cn_profiles.copy()
        cn_plot_profiles['Positions'] = cn_plot_profiles['Start']
    cn_plot = [cn_plot_profiles.pivot_table(columns=['Chr', 'Positions'], index='Node', values=cn_col) for cn_col in [j for j in cn_profiles.columns if ('cn' in j.lower()) or (j in ['A', 'B'])]]

    chr_data = cn_plot_profiles[['Chr', 'Positions']].drop_duplicates().sort_values(by=['Chr', 'Positions']).reset_index(drop=True)
    return cn_plot, chr_data

    
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
    return coords, edges_plot, y_vals


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

def process_cn_profile_comp_data(cn_profiles, cn_profiles2, root, plot_root, real_chr_lengths):
    if plot_root == False:
        cn_profiles = cn_profiles[cn_profiles.Node != root].copy()
    cn_profiles = cn_profiles.merge(cn_profiles2, on=['Node', 'Chr', 'Start', 'End'], suffixes=('_spice', '_other'))
    cn_profiles['cnA'] = cn_profiles['cnA_spice'] != cn_profiles['cnA_other']
    cn_profiles['cnB'] = cn_profiles['cnB_spice'] != cn_profiles['cnB_other']
    cn_profiles = cn_profiles.drop(['cnA_spice', 'cnB_spice', 'cnA_other', 'cnB_other'], axis=1)
    wrong = cn_profiles[(cn_profiles['cnA'] == True) | (cn_profiles['cnB'] == True)]
    if real_chr_lengths:
        cn_profiles['Positions'] = cn_profiles.apply((lambda x: range(x.Start, x.End, 1000000)), axis=1)
        cn_plot_profiles = cn_profiles.explode('Positions').reset_index(drop=True)
    else:
        cn_plot_profiles = cn_profiles.copy()
        cn_plot_profiles['Positions'] = cn_plot_profiles['Start']
    cn_plot = [cn_plot_profiles.pivot_table(columns=['Chr', 'Positions'], index='Node', values=cn_col) for cn_col in [j for j in cn_profiles.columns if ('cn' in j.lower()) or (j in ['A', 'B'])]]

    chr_data = cn_plot_profiles[['Chr', 'Positions']].drop_duplicates().sort_values(by=['Chr', 'Positions']).reset_index(drop=True).reset_index(drop=True)
    return cn_plot, chr_data, wrong

def add_cn_profile_comp(ax, cn_plot, leaves, root, chr_data, coords, cn_palette):
    sns.heatmap(cn_plot, cmap='Reds', vmax=1, vmin=0, ax=ax, cbar=False, linewidth=0, linecolor='black')
    ax.set_ylim(-.5, len(coords)+1.5)
    ax.set_xticks([])
    ax.yaxis.set_tick_params(labelleft=False)
    # ax.set_yticks([])
    ax.set_yticklabels([i.replace('sample_', '') if i in leaves else f"int_{i}" if ('internal' not in i) & (i != root) else i.replace('internal_', 'int').replace('internal', 'int') for i in list(cn_plot.index)])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for i in range(len(coords)):
        ax.axhline(y=i, color='white', linewidth=3)  ##ECECEC
    ax.vlines(x=list(chr_data[chr_data.Chr != chr_data.Chr.shift()].index), ymin=0, ymax=len(coords)+.5, color='black', linestyles='dotted', linewidth=2)

def process_cn_profiles(cn_profiles, chrs):
    cn_profiles['Node'] = cn_profiles['Node'].astype(str)
    if 'Allele' in cn_profiles.columns:
        idx_cols = [i for i in cn_profiles.columns if i not in ['Allele', 'CN']]
        cn_profiles['Allele'] = cn_profiles['Allele'].map({'A' : 'cnA', 'B' : 'cnB'})
        cn_profiles = cn_profiles.pivot_table(index=idx_cols, columns='Allele', values='CN').reset_index()
    if chrs != 'all':
        cn_profiles = cn_profiles[cn_profiles['Chr'].isin(chrs)].copy()
    return cn_profiles


def plot_cn_comparison(cn_out_tree, cn_profiles, cn_profiles2, root, leaf_order=None, cn_palette='RdBu_r', color_mapping={}, ax=None, plot_root=True, idx1=None, idx2=None, chrs='all', profile2_mapping=None, medicc=False, real_chr_lengths=True):
    if idx1 is not None:
        cn_profiles = cn_profiles[cn_profiles['Solution'] == idx1].copy()
        cn_out_tree = cn_out_tree[cn_out_tree['Solution'] == idx1].copy()
        
    if idx2 is not None:
        cn_profiles2 = cn_profiles2[cn_profiles2['Solution'] == idx2].copy()
        
    cn_profiles = process_cn_profiles(cn_profiles, chrs)
    cn_profiles2 = process_cn_profiles(cn_profiles2, chrs)

    cn_out_tree[['Parent', 'Child']] = cn_out_tree[['Parent', 'Child']].astype(str)
    root = str(root)
     
    parents = cn_out_tree['Parent'].unique()
    children = cn_out_tree['Child'].unique()
    leaves = children[~np.isin(children, parents)]
    
    if medicc:
        cn_profiles['Node'] = np.where(cn_profiles['Node'].isin(leaves), cn_profiles['Node'].astype(str), 'internal_' + cn_profiles['Node'].astype(str))
        cn_out_tree['Parent'] = 'internal_' + cn_out_tree['Parent'].astype(str)
        cn_out_tree['Child'] = np.where(cn_out_tree['Child'].isin(leaves), cn_out_tree['Child'].astype(str), 'internal_' + cn_out_tree['Child'].astype(str))
    
    if not (idx1 is not None and idx2 is not None):
        if profile2_mapping is None:
            cn_profiles2['Node'] = np.where(cn_profiles2['Node'].isin(leaves), cn_profiles2['Node'].astype(str), 'internal_' + cn_profiles2['Node'].astype(str))
        else:
            profile2_mapping = {v : str(k) if 'sample' in v else f'internal_{k}' for k, v in profile2_mapping.items()}
            cn_profiles2['Node'] = cn_profiles2['Node'].map(profile2_mapping)
    
    coords, edges_plot, y_vals = get_coord_df(cn_out_tree, root, leaves, leaf_order, color_mapping, plot_root)
    
    palette = dict(zip(coords['index'], coords['color']))
    cn_plot, chr_data, wrong = process_cn_profile_comp_data(cn_profiles, cn_profiles2, root, plot_root, real_chr_lengths)
    
    if plot_root:
        cn_index = list(y_vals.keys())
    else:
        cn_index = [i for i in y_vals.keys() if i != root]
    cn_plot = [cn_plot_df.reindex(cn_index) for cn_plot_df in cn_plot]
    
    allele_specific = True if len(cn_plot) > 1 else False

    sizey = int(cn_profiles.Node.nunique()*0.6) + 1
    if ax is None:
        if allele_specific:
            fig, ax = plt.subplots(1, 4, figsize=(30, sizey), width_ratios=[5, 0.2, 15, 15], sharey='row')
        else:
            fig, ax = plt.subplots(1, 3, figsize=(25, sizey), width_ratios=[6, 0.2, 15], sharey='row')
        
    ax[1].set_visible(False)
    
    # Top left = colorbar
    add_cn_legend(ax[0], cn_palette)
    
    # Top right - chrs
    add_chr_bar(ax[2], chr_data, 'Allele A')
    if allele_specific:
        add_chr_bar(ax[3], chr_data, 'Allele B')

    # Bottom left - Tree
    add_cn_tree(ax[0], edges_plot, coords, palette, root, leaves, plot_root)

    # Bottom right - CNs
    for i, cn_profiles in enumerate(cn_plot):
        add_cn_profile_comp(ax[i+2], cn_profiles, leaves, root, chr_data, coords, cn_palette)
    
    if idx1 is not None: 
        add = f' vs {idx2}' if idx2 is not None else ''
        ax[2].text(0, -1, f'Solution #{idx1}{add}', fontsize=18, ha='center', va='center', zorder=10)
        
    plt.subplots_adjust(wspace=0.08, hspace=-0.05);
    
    # return wrong