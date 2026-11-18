"""Plot SPICE copy number trees and profiles.

Both figures draw the tree on the left, with a star on each edge that has a whole genome doubling (WGD), and one panel
per allele on the right (or one for total copy number), with the chromosomes along the top:

- `plot_cn_tree` takes an edge table and a node table from `run_spice_bootstrap` (e.g. 'total_edges_multisol' and
  'total_nodes_multisol'), selects one solution and shows each node's copy numbers.
- `plot_cn_probabilities` also takes 'cn_probabilities', selects one topology (by default that of the top-ranked
  solution) and shows the probability of each copy number along the genome for each node below the root; leaves
  have their copy numbers with probability 1.
"""

from typing import Optional, Sequence, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import blended_transform_factory, offset_copy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

# Colormap of the copy number probabilities, from pale yellow (low) to dark blue (high)
event_prob_cmap = mcolors.LinearSegmentedColormap.from_list(
    'event_prob_cmap', ['#E8E79AFF', '#C2D89AFF', '#8CBF9AFF', '#5FA2A4FF', '#477B95FF', '#315B88FF', '#24396BFF',
                        '#191F40FF'], N=256)

# Fixed sizes shared by both figures, in inches. Heights: a row per node (`plot_cn_tree`) or per copy number value
# (`plot_cn_probabilities`), and the margins above and below the rows. Widths at `scale_x` = 1, by `allele_specific`:
# the tree and each copy number panel; the space for the node labels, between the two allele panels and the margins
# stay fixed when `scale_x` changes.
ROW_HEIGHT = 0.4
PROB_ROW_HEIGHT = 0.1
MARGIN_TOP = 0.5
MARGIN_BOTTOM = 0.45
TREE_WIDTH = {True: 3, False: 5}
PANEL_WIDTH = {True: 9, False: 12.4}
LABEL_SPACE = 1.7
PANEL_SPACE = 0.6
MARGIN_LEFT = 1
MARGIN_RIGHT = 0.3


def _figure_grid(n_rows: int, allele_specific: bool, scale_x: float, height: float, label_space: float = LABEL_SPACE,
                 hspace: float = 0) -> plt.GridSpec:
    """Create a figure with columns tree, label space and one panel per allele, and return its grid.

    `n_rows` rows share `height` inches (plus the margins), separated by `hspace` times their height.
    """
    widths = [TREE_WIDTH[allele_specific]*scale_x, label_space, PANEL_WIDTH[allele_specific]*scale_x]
    if allele_specific:
        widths += [PANEL_SPACE, PANEL_WIDTH[allele_specific]*scale_x]
    fig_width = sum(widths) + MARGIN_LEFT + MARGIN_RIGHT
    fig_height = height + MARGIN_TOP + MARGIN_BOTTOM
    fig = plt.figure(figsize=(fig_width, fig_height))
    return fig.add_gridspec(n_rows, len(widths), width_ratios=widths, wspace=0, hspace=hspace,
                            left=MARGIN_LEFT/fig_width, right=1 - MARGIN_RIGHT/fig_width,
                            top=1 - MARGIN_TOP/fig_height, bottom=MARGIN_BOTTOM/fig_height)


def plot_cn_tree(cn_out_tree: pd.DataFrame,
                cn_profiles: pd.DataFrame,
                leaf_order: Optional[Sequence[str]] = None,
                cn_palette: str = 'RdBu_r',
                color_mapping: Optional[dict] = None,
                plot_root: bool = True,
                solution: Optional[str] = None,
                chrs: Union[str, Sequence[int]] = 'all',
                real_chr_lengths: bool = True,
                max_cn: Optional[float] = None,
                scale_x: float = 1,
                scale_y: float = 1) -> None:
    """Plot a copy number tree with a heatmap of the copy numbers of each node.

    Parameters
    ----------
    cn_out_tree : pd.DataFrame
        Edge table, one row per edge (and solution), with 'Parent', 'Child' and 'WGD' (1 if the edge has a WGD), and
        'Solution' if `solution` is given. If `solution` is None, 'Parent' and 'Child' are cast to str in place.
    cn_profiles : pd.DataFrame
        Node table, one row per node and segment, with 'Node', 'Chr', 'Start', 'End' and the copy numbers in 'cnA' /
        'cnB' (allele-specific) or 'CN' (total copy number). A long table with 'Allele' ('A' / 'B') and 'CN' is also
        accepted. Needs 'Solution' if `solution` is given.
    leaf_order : sequence of str, optional
        Preferred order of the leaves from the bottom of the figure up; the tree topology takes precedence. By default
        the leaves are sorted by name.
    cn_palette : str
        Matplotlib colormap for the heatmap, centred on the diploid copy number.
    color_mapping : dict, optional
        Colour of each node's marker in the tree, {node: colour}; nodes not listed are white.
    plot_root : bool
        If True, the root gets its own row in the tree and the heatmap.
    solution : str, optional
        'Solution' to plot (e.g. 'm1'). If None, each table must hold a single solution (or have no 'Solution' column).
    chrs : 'all' or sequence of int
        Chromosomes to plot.
    real_chr_lengths : bool
        If True, segment widths in the heatmap are proportional to their length (1 Mb steps); otherwise each segment
        is one column.
    max_cn : float, optional
        Top of the colour scale. By default the highest copy number plotted.
    scale_x : float
        Factor for the width of the tree and the heatmaps; the spaces between them stay the same.
    scale_y : float
        Factor for the height of each row (`ROW_HEIGHT` inches).

    Returns
    -------
    None
        The figure is drawn on a new figure (`plt.gcf()`).
    """
    if color_mapping is None:
        color_mapping = {}
    if solution is not None:
        assert (solution in cn_out_tree['Solution'].values) and (solution in cn_profiles['Solution'].values), \
            f"Solution {solution!r} is not in both tables; available: {sorted(cn_out_tree['Solution'].unique())}"
        cn_profiles = cn_profiles[cn_profiles['Solution'] == solution].copy()
        cn_out_tree = cn_out_tree[cn_out_tree['Solution'] == solution].copy()
    else:
        for name, table in [('cn_out_tree', cn_out_tree), ('cn_profiles', cn_profiles)]:
            assert ('Solution' not in table.columns) or (table['Solution'].nunique() <= 1), \
                f'{name} has {table["Solution"].nunique()} solutions; choose one with `solution`'

    cn_profiles = process_cn_profiles(cn_profiles, chrs)
    if max_cn is None:
        max_cn = cn_profiles[[i for i in ['cnA', 'cnB', 'cn', 'CN'] if i in cn_profiles.columns]].values.flatten().max()

    cn_out_tree[['Parent', 'Child']] = cn_out_tree[['Parent', 'Child']].astype(str)

    parents = cn_out_tree['Parent'].unique()
    children = cn_out_tree['Child'].unique()
    leaves = children[~np.isin(children, parents)]
    roots = parents[~np.isin(parents, children)]
    assert len(roots) == 1, f'The edges must form a single tree, found {len(roots)} roots: {list(roots)}'
    root = roots[0]

    coords, edges_plot, y_vals = get_coord_df(cn_out_tree, root, leaves, leaf_order, color_mapping, plot_root)

    palette = dict(zip(coords['index'], coords['color']))
    cn_plot, chr_data = process_cn_profiles_data(cn_profiles, root, plot_root, real_chr_lengths)

    if plot_root:
        cn_index = list(y_vals.keys())
    else:
        cn_index = [i for i in y_vals.keys() if i != root]
    cn_plot = [cn_plot_df.reindex(cn_index) for cn_plot_df in cn_plot]

    allele_specific = len(cn_plot) > 1

    gs = _figure_grid(1, allele_specific, scale_x, len(cn_index)*ROW_HEIGHT*scale_y)
    tree_ax = plt.gcf().add_subplot(gs[0, 0])
    cn_axes = [plt.gcf().add_subplot(gs[0, col], sharey=tree_ax) for col in range(2, gs.ncols, 2)]

    # Top left - colorbar
    add_cn_legend(tree_ax, cn_palette, allele_specific, max_cn)

    # Top right - chrs
    name = 'Allele A' if allele_specific else 'Total Copy Number'
    add_chr_bar(cn_axes[0], chr_data, name)
    if allele_specific:
        add_chr_bar(cn_axes[1], chr_data, 'Allele B')

    # Bottom left - Tree
    add_cn_tree(tree_ax, edges_plot, coords, palette, root, leaves, cn_axes[0], plot_root)
    add_wgd(tree_ax, edges_plot, cn_out_tree)

    # Bottom right - CNs
    for cn_ax, cn_profiles in zip(cn_axes, cn_plot):
        add_cn_profiles(cn_ax, cn_profiles, chr_data, cn_palette, allele_specific, max_cn)

    if solution is not None:
        _add_title(tree_ax, f'Solution: #{solution}')


def add_wgd(ax: plt.Axes, coords: pd.DataFrame, cn_out_tree: pd.DataFrame) -> None:
    """Mark each edge that has a whole genome doubling (WGD) with a star at its midpoint.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes with the tree.
    coords : pd.DataFrame
        Edge coordinates from `get_coord_df` (`edges_plot`), with 'Child', 'x_Parent', 'y_Parent', 'x_Child' and
        'y_Child'.
    cn_out_tree : pd.DataFrame
        Edge table with 'Child' and 'WGD' (1 if the edge has a WGD).

    Returns
    -------
    None
    """
    wgd_branches = cn_out_tree[cn_out_tree['WGD'] == 1]['Child'].unique()
    wgd_coords = coords[coords['Child'].isin(wgd_branches)].copy()
    wgd_coords['x'] = (wgd_coords['x_Parent'] + wgd_coords['x_Child']) / 2
    wgd_coords['y'] = (wgd_coords['y_Parent'] + wgd_coords['y_Child']) / 2
    sns.scatterplot(x='x', y='y', data=wgd_coords, legend=False, s=350, marker='*', edgecolor='black', linewidth=1,
                    color='gold', zorder=2, ax=ax, alpha=1)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])


# Distance of the title and colour key from the left edge of the tree axes, in points
HEADER_X = -36


def _add_title(ax: plt.Axes, title: str) -> None:
    """Write `title` above the top left corner of the tree axes `ax`, starting `HEADER_X` points from its left edge."""
    ax.text(0, 1, title, fontsize=20, ha='left', va='bottom',
            transform=offset_copy(ax.transAxes, fig=ax.figure, x=HEADER_X, y=20, units='points'))


def _legend_axes(ax: plt.Axes, height: float) -> plt.Axes:
    """Return an inset axes, 0.2 inches wide and `height` inches high, below the title of the tree axes `ax`."""
    return inset_axes(ax, width=0.2, height=height, loc='upper left', bbox_to_anchor=(0, 1), borderpad=0,
                      bbox_transform=offset_copy(ax.transAxes, fig=ax.figure, x=HEADER_X, y=-30, units='points'))


def add_cn_legend(ax: plt.Axes, cn_palette: str, allele_specific: bool, max_cn: float) -> None:
    """Draw the copy number colour key as a column of squares in the top left corner of `ax`, below its title.

    The key runs from 0 to `max_cn` in steps of 1, coloured as in the heatmap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes with the tree; the key is drawn in an inset axes.
    cn_palette : str
        Matplotlib colormap of the heatmap.
    allele_specific : bool
        If True, the colormap is centred on 1 (copies per allele), otherwise on 2 (total copy number).
    max_cn : float
        Highest copy number in the key, and the top of the colour scale.

    Returns
    -------
    None
    """
    center = 1 if allele_specific else 2
    ax_cbar = _legend_axes(ax, 0.2*(max_cn+1))
    sns.heatmap(pd.DataFrame(np.arange(0, max_cn+1)), ax=ax_cbar, cmap=cn_palette, cbar=False, linewidths=2,
                linecolor='white', vmin=0, vmax=max_cn, center=center)
    ax_cbar.invert_yaxis()
    ax_cbar.set_xticks([])
    ax_cbar.yaxis.tick_right()
    ax_cbar.tick_params(axis='y', rotation=0, labelsize=12, length=0)
    ax_cbar.set_title('# Copies', fontsize=16, pad=10, loc='left')


def add_prob_legend(ax: plt.Axes) -> None:
    """Draw the copy number probability colour bar (0 to 1) in the top left corner of `ax`, below its title."""
    ax_cbar = _legend_axes(ax, 1)
    plt.colorbar(ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=event_prob_cmap), cax=ax_cbar, ticks=[0, 0.5, 1])
    ax_cbar.tick_params(labelsize=12, length=0)
    ax_cbar.set_title('Probability', fontsize=16, pad=10, loc='left')


def add_chr_bar(ax: plt.Axes, chr_data: pd.DataFrame, title: Optional[str] = None) -> None:
    """Draw a chromosome bar along the top of a copy number panel, shading alternate chromosomes and labelling each one.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of the top copy number panel; the bar is drawn in an inset axes, 0.2 inches high, along its top.
    chr_data : pd.DataFrame
        One row per heatmap column, with 'Chr' and 'Positions', from `process_cn_profiles_data`. A 'chr_code' column
        (alternating 0 / 1 per chromosome) is added in place.
    title : str, optional
        Panel title above the bar (e.g. 'Allele A').

    Returns
    -------
    None
    """
    ax_chr = inset_axes(ax, width='100%', height=0.2, loc='lower left', bbox_to_anchor=(0, 1, 1, 1),
                        bbox_transform=ax.transAxes, borderpad=0)
    chr_data['chr_code'] = (chr_data['Chr']-1) % 2
    chr_data_plot = chr_data['chr_code'].to_frame().T
    sns.heatmap(chr_data_plot, cmap='Greys_r', vmax=2, vmin=-1, ax=ax_chr, cbar=False, alpha=0.9)
    # Each chromosome's number in the middle of its band, white on the dark bands and black on the light ones. In a
    # run of narrow bands (under 3% of the bar), every other number is raised above the bar so that they don't overlap
    narrow_run = 0
    for c, cols in chr_data.groupby('Chr', sort=False).groups.items():
        narrow_run = narrow_run + 1 if len(cols) < 0.03*len(chr_data) else 0
        raised = narrow_run % 2 == 1
        ax_chr.text(x=(cols.min() + cols.max() + 1) / 2, y=0 if raised else 0.5, s=str(c), fontsize=12, ha='center',
                    va='bottom' if raised else 'center', color='white' if (c-1) % 2 == 0 and not raised else 'black')
    ax_chr.set_ylabel('')
    ax_chr.set_xlabel('')
    ax_chr.set_xticks([])
    ax_chr.set_yticks([])

    if title is not None:
        ax_chr.set_title(title, fontsize=16, pad=10)


def _node_label(node: str, leaves: np.ndarray) -> str:
    """Return the axis label of a node: leaves without 'sample_', other nodes unchanged."""
    if node in leaves:
        return node.replace('sample_', '')
    return node


def add_cn_tree(ax: plt.Axes, edges_plot: pd.DataFrame, coords: pd.DataFrame, palette: dict, root: str,
                leaves: np.ndarray, label_ax: plt.Axes, plot_root: bool = False, plot_node_names: bool = True,
                label_pad: float = 6) -> None:
    """Draw the tree: nodes as circles, edges as lines, and a faint guide line from each node to its copy number row.

    Node labels (see `_node_label`) end `label_pad` points to the left of the copy number panels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw the tree on.
    edges_plot : pd.DataFrame
        Edge coordinates from `get_coord_df`, with 'x_Parent', 'y_Parent', 'x_Child' and 'y_Child'.
    coords : pd.DataFrame
        Node coordinates from `get_coord_df`, with 'index' (node), 'x', 'y' and 'name'.
    palette : dict
        Colour of each node's circle, {node: colour}.
    root : str
        Name of the root node.
    leaves : numpy.ndarray
        Names of the leaves.
    label_ax : matplotlib.axes.Axes
        A copy number panel of the first column; the guide lines end at its left edge.
    plot_root : bool
        If False, the root gets no guide line or label.
    plot_node_names : bool
        If False, the nodes are not labelled.
    label_pad : float
        Space between the labels and the left edge of `label_ax`, in points.

    Returns
    -------
    None
    """
    sns.scatterplot(x='x', y='y', data=coords, legend=False, s=750, edgecolor='black', linewidth=2, hue=coords['index'],
                    palette=palette, zorder=3, ax=ax, alpha=1, clip_on=False)
    for _, row in edges_plot.iterrows():
        ax.plot([row.x_Parent, row.x_Child], [row.y_Parent, row.y_Child], color='black', linewidth=5, zorder=0)
    # x in axes coordinates of `label_ax` (0 = its left edge), y in data coordinates of the tree
    panel_edge = blended_transform_factory(label_ax.transAxes, ax.transData)
    labelled = coords if plot_root else coords[coords['index'] != root]
    for _, row in labelled.iterrows():
        ax.annotate('', xy=(0, row.y), xycoords=panel_edge, xytext=(row.x, row.y), textcoords='data', zorder=0,
                    arrowprops=dict(arrowstyle='-', color='grey', linewidth=8, alpha=0.15, shrinkA=0, shrinkB=0))
        if plot_node_names:
            ax.text(0, row.y, _node_label(row['name'], leaves), fontsize=16, ha='right', va='center',
                    transform=offset_copy(panel_edge, fig=ax.figure, x=-label_pad, units='points'))
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_xlim(coords['x'].min()-1, coords['x'].max()+0.25)
    ax.set_ylim(coords['y'].min()-.5, coords['y'].max()+.5)
    sns.despine(left=True, bottom=True, ax=ax)
    ax.spines['right'].set_color('white')


def add_cn_profiles(ax: plt.Axes, cn_plot: pd.DataFrame, chr_data: pd.DataFrame, cn_palette: str,
                    allele_specific: bool, max_cn: float) -> None:
    """Draw one heatmap of copy numbers, one row per node, with white lines between chromosomes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw the heatmap on.
    cn_plot : pd.DataFrame
        Copy numbers of one allele (or total copy number), one row per node in tree order and one column per
        ('Chr', 'Positions'), from `process_cn_profiles_data`.
    chr_data : pd.DataFrame
        One row per heatmap column, with 'Chr', from `process_cn_profiles_data`.
    cn_palette : str
        Matplotlib colormap of the heatmap.
    allele_specific : bool
        If True, the colormap is centred on 1 (copies per allele), otherwise on 2 (total copy number).
    max_cn : float
        Top of the colour scale.

    Returns
    -------
    None
    """
    ax.set_facecolor('white')
    center = 1 if allele_specific else 2
    sns.heatmap(cn_plot, cmap=cn_palette, vmax=max_cn, center=center, vmin=0, ax=ax, cbar=False, linewidth=0,
                linecolor='white')
    ax.set_ylim(0, len(cn_plot))
    ax.set_xticks([])
    ax.tick_params(axis='y', length=0, labelleft=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    for i in range(1, len(cn_plot)):
        ax.axhline(y=i, color='white', linewidth=3)
    for i in chr_data[chr_data.Chr != chr_data.Chr.shift()].index[1:]:
        ax.axvline(x=i, color='white', linewidth=2)


def process_cn_profiles_data(cn_profiles: pd.DataFrame, root: str, plot_root: bool,
                             real_chr_lengths: bool) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Turn the node table into one heatmap table per allele, with one column per genome position plotted.

    Parameters
    ----------
    cn_profiles : pd.DataFrame
        Node table from `process_cn_profiles`, with 'Node', 'Chr', 'Start', 'End' and the copy number columns. If
        `plot_root` and `real_chr_lengths` are True, a 'Positions' column is added in place.
    root : str
        Name of the root node.
    plot_root : bool
        If False, the root's rows are left out.
    real_chr_lengths : bool
        If True, each segment is split into 1 Mb steps from 'Start', so its width is proportional to its length (a
        segment shorter than 1 Mb is one column); otherwise each segment is one column.

    Returns
    -------
    cn_plot : list of pd.DataFrame
        One table per copy number column ('cnA' and 'cnB', or 'CN'), with one row per node and one column per
        ('Chr', 'Positions').
    chr_data : pd.DataFrame
        One row per heatmap column, with 'Chr' and 'Positions', in column order.
    """
    if not plot_root:
        cn_profiles = cn_profiles[cn_profiles.Node != root].copy()
    if real_chr_lengths:
        cn_profiles['Positions'] = cn_profiles.apply(lambda x: range(x.Start, x.End, 1000000), axis=1)
        cn_plot_profiles = cn_profiles.explode('Positions').reset_index(drop=True)
    else:
        cn_plot_profiles = cn_profiles.copy()
        cn_plot_profiles['Positions'] = cn_plot_profiles['Start']
    cn_cols = [j for j in ['cnA', 'cnB', 'cn', 'CN', 'A', 'B'] if j in cn_profiles.columns]
    cn_plot = [cn_plot_profiles.pivot_table(columns=['Chr', 'Positions'], index='Node', values=cn_col)
               for cn_col in cn_cols]

    chr_data = (cn_plot_profiles[['Chr', 'Positions']].drop_duplicates().sort_values(by=['Chr', 'Positions'])
                .reset_index(drop=True))
    return cn_plot, chr_data


def get_coord_df(cn_out_tree: pd.DataFrame, root: str, leaves: np.ndarray, leaf_order: Optional[Sequence[str]] = None,
                 color_mapping: Optional[dict] = None,
                 plot_root: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Compute the plot coordinates of the tree's nodes and edges.

    x comes from `get_x_order` and y (one row per node) from `get_y_order`. y is shifted by 0.5 so that each node
    lines up with the centre of its heatmap row.

    Parameters
    ----------
    cn_out_tree : pd.DataFrame
        Edge table with 'Parent' and 'Child' (str).
    root : str
        Name of the root node.
    leaves : numpy.ndarray
        Names of the leaves.
    leaf_order : sequence of str, optional
        Preferred order of the leaves from the bottom up (see `get_y_order`).
    color_mapping : dict, optional
        Colour of each node's circle, {node: colour}; nodes not listed are white.
    plot_root : bool
        If True, the root gets its own row (see `get_y_order`).

    Returns
    -------
    coords : pd.DataFrame
        One row per node, sorted by y, with 'index' (node), 'x', 'y', 'color' and 'name' (the node name, cut at the
        first '.' if it contains '_').
    edges_plot : pd.DataFrame
        One row per edge, with 'Parent', 'Child' and the coordinates of both ends ('x_Parent', 'y_Parent',
        'x_Child', 'y_Child').
    y_vals : dict
        Row of each node, {node: row}, before the 0.5 shift.
    """
    if color_mapping is None:
        color_mapping = {}
    x_vals = get_x_order(cn_out_tree, root, leaves)
    y_vals = get_y_order(cn_out_tree, root, leaves, leaf_order, plot_root)
    coords = pd.DataFrame({'x': x_vals, 'y': y_vals}).reset_index().sort_values(by='y').reset_index()
    coords['color'] = coords['index'].apply(lambda x: color_mapping[x] if x in color_mapping else '#FFFFFF')
    coords['name'] = np.where(coords['index'].astype(str).str.contains('_'),
                              coords['index'].astype(str).str.split('.').str[0], coords['index'].astype(str))
    edges_plot = (cn_out_tree[['Parent', 'Child']].astype(str)
                  .merge(coords, left_on='Parent', right_on='index')
                  .merge(coords, left_on='Child', right_on='index', suffixes=['_Parent', '_Child']))
    coords['y'] = coords['y'] + 0.5
    edges_plot['y_Parent'] = edges_plot['y_Parent'] + 0.5
    edges_plot['y_Child'] = edges_plot['y_Child'] + 0.5
    return coords, edges_plot, y_vals


def get_x_order(cn_out_tree: pd.DataFrame, root: str, leaves: np.ndarray) -> dict[str, float]:
    """Compute the x coordinate of each node, with all leaves aligned at x = 0.

    A node is placed by its position on the path from its deepest leaf to the root, as a fraction of that path
    (`calculate_fract_path`), scaled to the longest leaf-to-root path in the tree, so internal nodes are to the left
    (negative x). The root is then moved 1 further to the left.

    Parameters
    ----------
    cn_out_tree : pd.DataFrame
        Edge table with 'Parent' and 'Child' (str).
    root : str
        Name of the root node.
    leaves : numpy.ndarray
        Names of the leaves.

    Returns
    -------
    dict
        x coordinate of each node, {node: x}.
    """
    nodes = np.unique(cn_out_tree[['Parent', 'Child']].values.flatten())
    tree_edges = list(zip(cn_out_tree['Parent'], cn_out_tree['Child']))
    leaf_paths = {leaf: get_leaf_path(leaf, tree_edges, []) for leaf in leaves}
    max_leaf_path = len(max(leaf_paths.values(), key=len))
    x_vals = {str(node): -calculate_fract_path(node, max(get_leaf_descendants(node, tree_edges, leaves, []),
                                                         key=lambda c: len(leaf_paths[c]), default=node),
                                               leaf_paths)*max_leaf_path
              for node in nodes}
    x_vals[str(root)] = x_vals[str(root)]-1
    return x_vals


def get_leaf_path(leaf: str, tree_edges: list[tuple], path: list = []) -> list:
    """Return the path from `leaf` up to the root, [leaf, parent, ..., root], appended to `path`."""
    path.append(leaf)
    parent = [i[0] for i in tree_edges if i[1] == leaf]
    if len(parent) == 0:
        return path
    else:
        path = get_leaf_path(parent[0], tree_edges, path)
        return path


def get_leaf_descendants(node: str, tree_edges: list[tuple], leaves: np.ndarray, children: list = []) -> list:
    """Return the leaves below `node` (depth first), appended to `children`."""
    node_childs = [i[1] for i in tree_edges if i[0] == node]
    if len(node_childs) == 0:
        return children
    else:
        for child in node_childs:
            if child in leaves:
                children.append(child)
            children = get_leaf_descendants(child, tree_edges, leaves, children)
        return children


# Fraction of the path from `leaf` up to the root that runs from `leaf` to `node` (both included); 0 if `node` is `leaf`
calculate_fract_path = (lambda node, leaf, leaf_paths:
                        0 if leaf == node
                        else len(leaf_paths[leaf][:leaf_paths[leaf].index(node)+1])/len(leaf_paths[leaf]))


def get_y_order(cn_out_tree: pd.DataFrame, root: str, leaves: np.ndarray, leaf_order: Optional[Sequence[str]] = None,
                plot_root: bool = False) -> dict[str, int]:
    """Compute the row (y coordinate) of each node, so that every node has its own row.

    Starting from the root's child, each node is placed between its children's subtrees: the children are sorted by
    the mean rank of their leaves in `leaf_order`, the first half goes below the node and the rest above it
    (`get_splits`).

    Parameters
    ----------
    cn_out_tree : pd.DataFrame
        Edge table with 'Parent' and 'Child' (str). The root must have a single child.
    root : str
        Name of the root node.
    leaves : numpy.ndarray
        Names of the leaves.
    leaf_order : sequence of str, optional
        Preferred order of the leaves from the bottom up; the tree topology takes precedence. By default the leaves are
        sorted by name.
    plot_root : bool
        If True, the root gets its own row just above its child; otherwise it shares its child's row.

    Returns
    -------
    dict
        Row of each node, {node: row}, from 0 at the bottom.
    """
    if leaf_order is None:
        leaf_order = sorted(leaves)
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


def get_splits(node: tuple[str, int], tree_edges: list[tuple], leaf_order: dict[str, int],
               leaves: np.ndarray) -> list[str]:
    """Return the subtree of `node` (name, number of descendants) in row order, half the child subtrees below it."""
    if node[1] == 0:
        return [node[0]]
    else:
        splits = sorted([(child, len(get_children(child, tree_edges, [])))
                         for child in get_node_children(node[0], tree_edges)],
                        key=lambda x: get_split_order(x[0], tree_edges, leaf_order, leaves))
        left_split = process_split(splits[:len(splits)//2], tree_edges, leaf_order, leaves)
        right_split = process_split(splits[len(splits)//2:], tree_edges, leaf_order, leaves)
    return left_split + [node[0]] + right_split


def process_split(split: list[tuple[str, int]], tree_edges: list[tuple], leaf_order: dict[str, int],
                  leaves: np.ndarray) -> list[str]:
    """Return the row orders of the subtrees in `split` (from `get_splits`), one after the other."""
    split_p = []
    for x in split:
        split_p.extend(get_splits(x, tree_edges, leaf_order, leaves))
    return split_p


def get_split_order(node: str, tree_edges: list[tuple], leaf_order: dict[str, int], leaves: np.ndarray) -> float:
    """Return the mean rank in `leaf_order` of the leaves below `node`, or its own rank if it is a leaf."""
    leaves = get_leaf_descendants(node, tree_edges, leaves, [])
    if len(leaves) == 0:
        return leaf_order[node]
    else:
        return sum(leaf_order[leaf] for leaf in leaves)/len(leaves)


def get_children(node: str, tree_edges: list[tuple], children: list = []) -> list:
    """Return all descendants of `node` (depth first), appended to `children`."""
    node_childs = [i[1] for i in tree_edges if i[0] == node]
    if len(node_childs) == 0:
        return children
    else:
        for child in node_childs:
            children.append(child)
            children = get_children(child, tree_edges, children)
        return children


# Direct children of `node`
get_node_children = (lambda node, tree_edges: [i[1] for i in tree_edges if i[0] == node])


def process_cn_profiles(cn_profiles: pd.DataFrame, chrs: Union[str, Sequence[int]]) -> pd.DataFrame:
    """Bring the node table into wide format (one copy number column per allele) and keep the chromosomes to plot.

    Parameters
    ----------
    cn_profiles : pd.DataFrame
        Node table with 'Node', 'Chr', 'Start', 'End' and either wide copy number columns ('cnA' / 'cnB' or 'CN') or
        long ones ('Allele' as 'A' / 'B', and 'CN'). 'Node' is cast to str in place, and for a long table 'Allele' is
        relabelled 'cnA' / 'cnB' in place.
    chrs : 'all' or sequence of int
        Chromosomes to keep.

    Returns
    -------
    pd.DataFrame
        The node table with 'cnA' / 'cnB' (or 'CN') columns, restricted to `chrs`.
    """
    cn_profiles['Node'] = cn_profiles['Node'].astype(str)
    if 'Allele' in cn_profiles.columns:
        idx_cols = [i for i in cn_profiles.columns if i not in ['Allele', 'CN']]
        cn_profiles['Allele'] = cn_profiles['Allele'].map({'A': 'cnA', 'B': 'cnB'})
        cn_profiles = cn_profiles.pivot_table(index=idx_cols, columns='Allele', values='CN').reset_index()
    if chrs != 'all':
        cn_profiles = cn_profiles[cn_profiles['Chr'].isin(chrs)].copy()
    return cn_profiles


def plot_cn_probabilities(cn_out_tree: pd.DataFrame,
                          cn_profiles: pd.DataFrame,
                          cn_probabilities: pd.DataFrame,
                          topology: Optional[int] = None,
                          leaf_order: Optional[Sequence[str]] = None,
                          color_mapping: Optional[dict] = None,
                          chrs: Union[str, Sequence[int]] = 'all',
                          real_chr_lengths: bool = True,
                          plot_node_names: bool = True,
                          scale_x: float = 1,
                          scale_y: float = 1) -> None:
    """Plot a copy number tree with the probability of each copy number of each node.

    Parameters
    ----------
    cn_out_tree : pd.DataFrame
        Edge table, one row per edge (and solution), with 'tree_id', 'Parent', 'Child' and 'WGD' (1 if the edge has a
        WGD).
    cn_profiles : pd.DataFrame
        Node table with 'tree_id', 'Node', 'Chr', 'Start', 'End' and the copy numbers in 'cnA' / 'cnB'
        (allele-specific) or 'CN' (total copy number). Only the leaves are used.
    cn_probabilities : pd.DataFrame
        Copy number probabilities from `run_spice_bootstrap` ('cn_probabilities'), one row per topology, node, segment
        (and allele) and copy number.
    topology : int, optional
        'tree_id' to plot. By default the topology of the top-ranked solution (the lowest 'tree_id' if several are
        tied), or the only one if `cn_probabilities` has no 'top_ranked_solution' column.
    leaf_order : sequence of str, optional
        Preferred order of the leaves from the bottom of the figure up; the tree topology takes precedence. By default
        the leaves are sorted by name.
    color_mapping : dict, optional
        Colour of each node's marker in the tree, {node: colour}; nodes not listed are white.
    chrs : 'all' or sequence of int
        Chromosomes to plot.
    real_chr_lengths : bool
        If True, segment widths are proportional to their length (1 Mb steps); otherwise each segment is one column.
    plot_node_names : bool
        If False, the nodes are not labelled.
    scale_x : float
        Factor for the width of the tree and the copy number panels; the spaces between them stay the same.
    scale_y : float
        Factor for the height of each row (`PROB_ROW_HEIGHT` inches per copy number value, so rows are taller when the
        copy numbers are higher).

    Returns
    -------
    None
        The figure is drawn on a new figure (`plt.gcf()`).
    """
    if topology is None:
        if 'top_ranked_solution' in cn_probabilities.columns:
            topology = cn_probabilities.loc[cn_probabilities['top_ranked_solution'], 'tree_id'].min()
        else:
            assert cn_probabilities['tree_id'].nunique() == 1, \
                ("cn_probabilities has no 'top_ranked_solution' column and several topologies; "
                 "choose one with `topology`")
            topology = cn_probabilities['tree_id'].iloc[0]
    assert topology in cn_probabilities['tree_id'].values, \
        f"Topology {topology} is not in cn_probabilities; available: {sorted(cn_probabilities['tree_id'].unique())}"
    cn_probabilities = cn_probabilities[cn_probabilities['tree_id'] == topology].astype({'Node': str})
    cn_profiles = cn_profiles[cn_profiles['tree_id'] == topology].astype({'Node': str})
    cn_out_tree = (cn_out_tree[cn_out_tree['tree_id'] == topology][['Parent', 'Child', 'WGD']].drop_duplicates()
                   .astype({'Parent': str, 'Child': str}))
    parents = cn_out_tree['Parent'].unique()
    children = cn_out_tree['Child'].unique()
    root = str(get_root(parents, children))
    leaves = get_leaves(parents, children)
    cn_all_profiles = concat_cn_dfs(cn_profiles, cn_probabilities, leaves)
    if chrs != 'all':
        cn_all_profiles = cn_all_profiles[cn_all_profiles['Chr'].isin(chrs)]
    cn_values = np.arange(0, cn_all_profiles['CN'].max() + 1).astype(int)

    # Add figure
    allele_specific = cn_all_profiles.Allele.nunique() == 2
    n_nodes = cn_all_profiles.Node.nunique()
    # Rows are separated by `hspace` times their height
    hspace = 0.07
    row_height = len(cn_values)*PROB_ROW_HEIGHT*scale_y
    gs = _figure_grid(n_nodes, allele_specific, scale_x, row_height*(n_nodes + (n_nodes-1)*hspace),
                      LABEL_SPACE if plot_node_names else 0.3, hspace)
    fig = plt.gcf()

    # Bottom left - Tree
    tree_ax = fig.add_subplot(gs[:, 0])
    coords, edges_plot, y_vals = get_coord_df(cn_out_tree, root, leaves, leaf_order, color_mapping, False)
    cn_index = [i for i in y_vals if i != root]
    palette = dict(zip(coords['index'], coords['color']))
    alleles = sorted(cn_all_profiles.Allele.unique())
    cn_ax = {(allele, node): fig.add_subplot(gs[node_idx, 2 + 2*allele_idx])
             for allele_idx, allele in enumerate(alleles) for node_idx, node in enumerate(cn_index[::-1])}
    # Labels clear of the copy number ticks on the left of each panel
    add_cn_tree(tree_ax, edges_plot, coords, palette, root, leaves, cn_ax[(alleles[0], cn_index[-1])], False,
                plot_node_names, label_pad=20)
    add_wgd(tree_ax, edges_plot, cn_out_tree)
    _add_title(tree_ax, f'Topology: #{topology}')
    add_prob_legend(tree_ax)

    # Bottom right - CNs
    cn_plot_data = {i: process_cn_probabilities_data(j, real_chr_lengths, cn_values[::-1])
                    for i, j in cn_all_profiles.groupby(['Allele', 'Node'])}
    add_cn_prob_profiles(cn_ax, cn_plot_data, cn_values)

    # Top right - Chromosomes
    chr_data = {allele: cn_plot_data[(allele, children[0])].columns.T.to_frame().reset_index(drop=True)
                for allele in alleles}
    for allele in alleles:
        title = 'Allele ' + allele[-1] if allele_specific else 'Total Copy Number'
        add_chr_bar(cn_ax[(allele, cn_index[-1])], chr_data[allele], title)


# Root of the tree: the parent that is no node's child
get_root = (lambda parents, children: parents[~np.isin(parents, children)][0])
# Leaves of the tree: the children that are no node's parent
get_leaves = (lambda parents, children: children[~np.isin(children, parents)])


def concat_cn_dfs(cn_profiles: pd.DataFrame, cn_probabilities: pd.DataFrame, leaves: np.ndarray) -> pd.DataFrame:
    """Add the leaves' copy numbers, with probability 1, to the copy number probabilities of the other nodes.

    Parameters
    ----------
    cn_profiles : pd.DataFrame
        Node table of one topology, with 'tree_id', 'Node', 'Chr', 'Start', 'End' and 'cnA' / 'cnB' or 'CN'.
    cn_probabilities : pd.DataFrame
        Copy number probabilities of the same topology. Without 'Allele' (total copy number), every row gets allele
        'A'.
    leaves : numpy.ndarray
        Names of the leaves.

    Returns
    -------
    pd.DataFrame
        One row per node, segment, allele and copy number, with its 'CN_probability'.
    """
    cn_cols = [i for i in ['cnA', 'cnB', 'cn', 'CN'] if i in cn_profiles.columns]
    cn_leaf_profiles = (cn_profiles[cn_profiles['Node'].isin(leaves)]
                        [['tree_id', 'Node', 'Chr', 'Start', 'End'] + cn_cols].drop_duplicates())
    # Melted into 'value' first: pandas does not allow value_name to be one of the melted columns ('CN')
    cn_leaf_profiles = (pd.melt(cn_leaf_profiles, id_vars=['tree_id', 'Node', 'Chr', 'Start', 'End'],
                                value_vars=cn_cols, var_name='Allele', value_name='value')
                        .rename(columns={'value': 'CN'}))
    cn_leaf_profiles['Allele'] = cn_leaf_profiles['Allele'].map({'cnA': 'A', 'cnB': 'B', 'cn': 'A', 'CN': 'A'})
    cn_leaf_profiles['CN_probability'] = 1
    if 'Allele' not in cn_probabilities.columns:
        cn_probabilities = cn_probabilities.assign(Allele='A')
    cn_all_profiles = pd.concat([cn_probabilities, cn_leaf_profiles], axis=0).reset_index(drop=True)
    return cn_all_profiles


def process_cn_probabilities_data(cn_profiles: pd.DataFrame, real_chr_lengths: bool,
                                  cn_values: np.ndarray) -> pd.DataFrame:
    """Turn the copy number probabilities of one node and allele into a heatmap table.

    Parameters
    ----------
    cn_profiles : pd.DataFrame
        Rows of one node and allele from `concat_cn_dfs`, with 'Chr', 'Start', 'End', 'CN' and 'CN_probability'.
    real_chr_lengths : bool
        If True, each segment is split into 1 Mb steps from 'Start', so its width is proportional to its length (a
        segment shorter than 1 Mb is one column); otherwise each segment is one column.
    cn_values : numpy.ndarray
        Copy numbers to show, in row order.

    Returns
    -------
    pd.DataFrame
        Probability of each copy number in `cn_values` (rows) at each ('Chr', 'Positions') (columns); 0 where the
        copy number has no probability.
    """
    if real_chr_lengths:
        cn_profiles['Positions'] = cn_profiles.apply((lambda x: range(x.Start, x.End, 1000000)), axis=1)
        cn_plot_profiles = cn_profiles.explode('Positions').reset_index(drop=True)
    else:
        cn_plot_profiles = cn_profiles.copy()
        cn_plot_profiles['Positions'] = cn_plot_profiles['Start']
    cn_plot = (cn_plot_profiles.pivot_table(columns=['Chr', 'Positions'], index='CN', values='CN_probability')
               .reindex(cn_values).fillna(0))
    return cn_plot


def add_cn_prob_profiles(cn_ax: dict[tuple[str, str], plt.Axes], cn_plot_data: dict[tuple[str, str], pd.DataFrame],
                         cn_values: np.ndarray) -> None:
    """Draw the heatmap of `cn_plot_data` on the axes in `cn_ax` with the same (allele, node) key, with white lines
    between chromosomes."""
    for data_idx, ax in cn_ax.items():
        sns.heatmap(cn_plot_data[data_idx].replace(0, np.nan), cmap=event_prob_cmap, vmax=1, vmin=0, ax=ax, cbar=False,
                    linewidth=0, linecolor='white')
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_yticks(ticks=[i+0.5 for i in list(cn_values)], labels=list(cn_values[::-1]), rotation=0, fontsize=10)
        sns.despine(left=False, bottom=False, ax=ax)
        chrs = cn_plot_data[data_idx].columns.get_level_values('Chr')
        for i in np.flatnonzero(chrs[1:] != chrs[:-1]) + 1:
            ax.axvline(x=i, color='white', linewidth=2)
