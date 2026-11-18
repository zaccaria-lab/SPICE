"""Pre-processing of the input leaf copy number profiles into the SPICE ILP input.

`process_spice_input` is the entry point, called by `run_spice_bootstrap`. It takes the leaf profiles as a long
DataFrame (one row per node, chromosome, allele and bin) and:

1. In total copy number mode (`allele_specific=False`), sums the two alleles, or labels a single copy number as
   allele 'A'.
2. Optionally phases the alleles (`phase_profiles`, allele-specific mode only).
3. Merges consecutive bins whose copy number is identical in every leaf into segments
   (`collapse_identical_consec_bins`), using breakpoints shared by both alleles.
4. Flags small segments for removal, either below `min_seg_size` (`filter_segments_segsize`) or outside the largest
   `max_n_segs` (`filter_segments_nsegs`, `get_best_filter_segs`). The largest segment of each chromosome is always
   kept, and `keep_consec` also keeps small segments so that every flagged segment has a neighbour in the ILP
   (`get_ilp_segs` on the `max_n_segs` path). The flagged segments are filled back in after the ILP
   (`fill_removed_segments`).
5. Merges the remaining segments again, per allele, and clips copy numbers to `max_cn` (`focal_max_cn` for the
   segments flagged for removal) with `clip_cns`.
6. Maps node names to integer IDs (`convert_node_names`, `rename_internal_nodes`): internal nodes first (the root is
   node 0), then the leaves.

Leaf profiles are returned in the ILP format {leaf: {chr: {allele: [cn per segment]}}}, with alleles keyed 0 and 1
(only 0 in total copy number mode). The position mapping links each original bin position (`origPos`) to its merged
(`mergePos`), filtered (`filteredPos`) and final ILP (`finalPos`) segment.
"""

import logging
import re
from typing import Optional, Sequence, Union

import pandas as pd
import numpy as np

from spice.libs.preprocess.phasecns import phase_profiles
from spice.libs.utils.logging import initialize_logger
from spice.libs.utils.conversion import cn_df_to_dict


def process_spice_input(leaves_df: pd.DataFrame, max_cn: Optional[int], phase_alleles: bool, allele_specific: bool,
                        leaf_dependent_phasing: bool, min_seg_size: float, max_n_segs: Optional[int],
                        keep_consec_segs: bool, focal_max_cn: Optional[int],
                        fixed_topology: Union[bool, Sequence[tuple], str], logger: Optional[logging.Logger] = None
                        ) -> tuple[dict, dict, dict, pd.DataFrame, pd.DataFrame]:
    """Process the input leaf copy number profiles into the inputs of the tree bootstrapping and the ILP.

    See the module docstring for the steps.

    Parameters
    ----------
    leaves_df : pd.DataFrame
        Leaf copy number profiles, one row per node, chromosome, allele and bin, with columns 'Node', 'Chr', 'Start',
        'End', 'Allele' ('A' or 'B', with both alleles of each bin) and 'CN'. In total copy number mode, 'Allele' may
        be left out; if given, alleles A and B are summed. Every leaf must have the same bins. Other columns are ignored
        with a warning. Not modified.
    max_cn : int or None
        Maximum copy number in the ILP (interval model); higher copy numbers are clipped to it. If None, no copy
        numbers are clipped, and `focal_max_cn` is ignored.
    phase_alleles : bool
        Whether to phase the alleles before running SPICE (allele-specific mode only; a warning is logged otherwise).
    allele_specific : bool
        If True, each allele's copy number is modelled separately; if False, total copy number is modelled and the
        alleles are summed.
    leaf_dependent_phasing : bool
        Whether a copy number segment is phased the same way in all leaves (only used if `phase_alleles` is True).
    min_seg_size : float
        Minimum segment size for the ILP; smaller segments are flagged for the focal evolutionary model. Ignored (with
        a warning if it is above 0) when `max_n_segs` is given.
    max_n_segs : int or None
        Maximum number of segments in the ILP: the largest `max_n_segs` segments are kept and the rest flagged for
        the focal evolutionary model. If None, segments are filtered by `min_seg_size`.
    keep_consec_segs : bool
        Whether to also keep runs of consecutive small segments in the ILP, rather than flagging them all.
    focal_max_cn : int or None
        Maximum copy number for the segments flagged for the focal evolutionary model. If None, `max_cn` is used.
    fixed_topology : bool, sequence of tuple or str
        False if the topology is inferred; otherwise the tree edges as (parent, child) node names, or a string of
        them. Used to choose the integer node IDs.
    logger : logging.Logger, optional
        Logger for progress messages. If None, a new SPICE logger is created with verbose output.

    Returns
    -------
    leaves_segments_dict : dict
        Leaf profiles for the tree bootstrapping, {leaf: {chr: {allele: [cn per segment]}}}, after merging
        identical consecutive bins but before filtering small segments.
    spice_input_dict : dict
        Leaf profiles for the ILP, {leaf: {chr: {allele: [cn per segment]}}}, after filtering, merging and clipping.
    node_name_mapping : dict
        Integer node ID for each node name, from `convert_node_names`.
    removed_seg_nodes : pd.DataFrame
        All segments after the first merge, one row per node, chromosome, segment and allele, with a 'filter_seg'
        column that is True for segments flagged for the focal evolutionary model. If `max_cn` is given, copy numbers
        are clipped to `focal_max_cn`.
    final_mapping : pd.DataFrame
        One row per original bin position, chromosome and allele, linking 'origPos' to 'mergePos', 'filteredPos'
        (NaN for flagged segments) and 'finalPos' (the ILP segment), with the bin's 'Start' and 'End'.
    """
    # Initialise logger
    if logger is None:
        logger, _ = initialize_logger(True, None)
    # Check for correct columns (ignore extra ones, including any 'Pos': the bins are numbered from 'Start' below)
    required_cols = ['Node', 'Chr', 'Start', 'End', 'CN'] + (['Allele'] if allele_specific else [])
    missing_cols = [i for i in required_cols if i not in leaves_df.columns]
    assert len(missing_cols) == 0, f'leaves_df is missing required columns: {missing_cols}'
    extra_cols = [i for i in leaves_df.columns if i not in required_cols + ['Allele']]
    if len(extra_cols) > 0:
        logger.warning(f'Ignoring columns of leaves_df that SPICE does not use: {extra_cols}')
    # Sort input dataframe
    leaves = (leaves_df
              .drop(columns='Pos', errors='ignore')
              .sort_values(by=['Node', 'Chr'] + (['Allele'] if 'Allele' in leaves_df.columns else []) + ['Start']))
    # Number each chromosome's bins from 1 in genome order; the leaves are lined up by these numbers, so they must
    # have the same bins
    pos_groups = ['Node', 'Chr'] + (['Allele'] if 'Allele' in leaves.columns else [])
    leaves['Pos'] = leaves.groupby(pos_groups)['Start'].rank(method='first').astype(int)
    n_leaves_per_bin = leaves.groupby(pos_groups[1:] + ['Start', 'End'])['Node'].nunique()
    missing_bins = n_leaves_per_bin.index[n_leaves_per_bin < leaves['Node'].nunique()]
    assert len(missing_bins) == 0, (f'Every leaf must have the same bins: {len(missing_bins)} bins are missing from '
                                    f'some leaves, e.g. {list(missing_bins[:3])} ({", ".join(missing_bins.names)})')
    # Process allele specific input
    leaves = standardise_alleles(leaves, allele_specific, 'leaves_df', logger)
    # Phase alleles
    if phase_alleles:
        if allele_specific:
            logger.info(f"Phasing alleles : {'leaf_dependent' if leaf_dependent_phasing else 'leaf_independent'}")
            leaves = phase_profiles(leaves, leaf_dependent=leaf_dependent_phasing)
        else:
            logger.warning('Phasing alleles not supported for non-allele specific mode. Ignoring phasing parameter.')
    # Collapse CN bins to segments
    logger.trace('Collapsing CN bins to segments.')
    original_genome_locs = (leaves[['Chr', 'Allele', 'Start', 'End', 'Pos']]
                            .drop_duplicates()
                            .sort_values(by=['Chr', 'Allele', 'Start', 'End'])
                            .rename(columns={'Pos': 'origPos'}))
    leaves_segments, pos_mapping = collapse_identical_consec_bins(leaves, allele_specific=False)
    pos_mapping = pos_mapping.rename(columns={'Pos': 'origPos',
                                              'group': 'mergePos'})
    # Separate small and large segments (based on min_seg_size)
    leaves_segments['seg_size'] = leaves_segments['End'] - leaves_segments['Start']
    consec_msg = ', plus any consecutive segments' if keep_consec_segs else ''
    if max_n_segs is not None:
        if min_seg_size > 0:
            logger.warning(f'Both max_n_segs and min_seg_size are specified. Filtering segments based on max_n_segs : '
                           f'the largest {max_n_segs} segments will be selected{consec_msg}.')
        else:
            logger.trace(f'Selecting the largest {max_n_segs} segments{consec_msg}.')
        filtered_leaves, removed_seg_nodes, filt_pos_mapping = filter_segments_nsegs(leaves_segments, pos_mapping,
                                                                                     n_segs=max_n_segs, logger=logger,
                                                                                     keep_consec=keep_consec_segs)
    else:
        if min_seg_size > 0:
            logger.trace(f'Selecting segments above a min_seg_size of {base_converter(min_seg_size)}{consec_msg}.')
        else:
            logger.trace('No min_seg_size or maximum number of segments specified. All segments will be kept.')
        filtered_leaves, removed_seg_nodes, filt_pos_mapping = filter_segments_segsize(leaves_segments, pos_mapping,
                                                                                       min_seg_size=min_seg_size,
                                                                                       logger=logger,
                                                                                       keep_consec=keep_consec_segs)
    # Rejoin large segments
    spice_input_nodes, final_mapping = collapse_identical_consec_bins(filtered_leaves, allele_specific=True)
    # Clip CNs of spice_input to max_cn and focal_max_cn
    if max_cn is not None:
        current_max_cn = spice_input_nodes.CN.max()
        prop_above_threshold = (spice_input_nodes.CN > max_cn).sum() / len(spice_input_nodes)
        if max_cn < current_max_cn:
            logger.warning(f'Clipping ILP model CN profiles to maximum value of {max_cn}. Previous maximum value was '
                        f'{current_max_cn}. {prop_above_threshold:.2%} of input CN values were above the threshold.')
        spice_input_nodes = clip_cns(spice_input_nodes, max_cn=max_cn)

        focal_max_cn = focal_max_cn if focal_max_cn is not None else max_cn
        current_focal_max_cn = removed_seg_nodes.CN.max()
        focal_prop_above_threshold = (removed_seg_nodes.CN > focal_max_cn).sum() / len(removed_seg_nodes)
        if focal_max_cn < current_focal_max_cn:
            logger.warning(f'Clipping focal evolutionary model CN profiles to maximum value of {focal_max_cn}. '
                           f'Previous maximum value was {current_focal_max_cn}. {focal_prop_above_threshold:.2%} of '
                           'input CN values were above the threshold.')
        removed_seg_nodes = clip_cns(removed_seg_nodes, max_cn=focal_max_cn)
    # Process mapping of original positions to final positions
    final_mapping = final_mapping.rename(columns={'Pos': 'filteredPos',
                                                  'group': 'finalPos'})
    final_mapping = (original_genome_locs
                     .merge(filt_pos_mapping, on=['Chr', 'Allele', 'origPos'], how='left')
                     .merge(final_mapping, on=['Chr', 'Allele', 'filteredPos'], how='left')
                     .astype(int, errors='ignore'))
    # Convert bootstrap leaves input to dictionary (bootstrap leaves are not filtered to remove small segments).
    node_name_mapping = convert_node_names(leaves['Node'].unique(), fixed_topology)
    leaves_segments_dict = cn_df_to_dict(leaves_segments, node_name_mapping, allele_specific)
    # Convert ILP leaves input to dictionary
    spice_input_dict = cn_df_to_dict(spice_input_nodes, node_name_mapping, allele_specific)
    test_node = list(spice_input_dict.keys())[0]
    logger.info(f'Processed input CN profiles. {len(spice_input_dict)} leaves with '
                f"{len(final_mapping[['Chr', 'Allele', 'finalPos']].dropna().drop_duplicates())} final segments")
    if allele_specific:
        chr_lengths = [f'{ch} : [A: {len(spice_input_dict[test_node][ch][0])} '
                       f'B: {len(spice_input_dict[test_node][ch][1])}]'
                       for ch in spice_input_dict[test_node]]
    else:
        chr_lengths = [f'{ch} : {len(spice_input_dict[test_node][ch][0])}'
                       for ch in spice_input_dict[test_node]]
    logger.trace(f"Chromosome lengths (ILP input): ({', '.join(chr_lengths)})")
    return leaves_segments_dict, spice_input_dict, node_name_mapping, removed_seg_nodes, final_mapping


def standardise_alleles(df: pd.DataFrame, allele_specific: bool, name: str, logger: logging.Logger) -> pd.DataFrame:
    """Check the 'Allele' column of a copy number table and match it to the mode.

    In allele-specific mode, 'Allele' is required. In total copy number mode, the alleles are summed if 'Allele' is
    given, otherwise the single copy number is used; either way the result is labelled allele 'A'. If given,
    'Allele' must have the values 'A' and 'B'.

    Parameters
    ----------
    df : pd.DataFrame
        Copy number table with 'Node', 'Chr', 'Start', 'End', 'CN', optionally 'Pos', and 'Allele' (required if
        `allele_specific`). Not modified.
    allele_specific : bool
        If True, each allele's copy number is modelled separately; if False, total copy number is modelled
        throughout the tree.
    name : str
        Name of the table, for messages.
    logger : logging.Logger
        Logger for the warning when alleles are summed.

    Returns
    -------
    pd.DataFrame
        `df` with an 'Allele' column; in total copy number mode, one row per node and bin, with allele 'A'.
    """
    assert (not allele_specific) or ('Allele' in df.columns), f"{name} must have an 'Allele' column"
    if 'Allele' in df.columns:
        alleles = set(df['Allele'].unique())
        assert alleles == {'A', 'B'}, f"'Allele' must have the values 'A' and 'B', found {sorted(alleles, key=str)}"
    if allele_specific:
        return df
    if 'Allele' in df.columns:
        logger.warning(f'Allele specific mode is turned off, but {name} contains multiple alleles. '
                       'Condensing to total copy numbers.')
        keys = [i for i in ['Node', 'Chr', 'Pos', 'Start', 'End'] if i in df.columns]
        return df.groupby(keys).agg({'CN': 'sum'}).reset_index().assign(Allele='A')
    # If total copy number we just assign all to 'A' to match ILP input format.
    return df.assign(Allele='A')


def base_converter(b: float) -> str:
    """Format a size in bases for log messages, e.g. 50000.0 -> '50.0kb'.

    Parameters
    ----------
    b : float
        Size in bases.

    Returns
    -------
    str
        The size in mb if it is at least 1,000,000, in kb if it is at least 1,000, otherwise in b.
    """
    if b >= 1000000:
        return '{}mb'.format(b / 1000000)
    if b >= 1000:
        return '{}kb'.format(b / 1000)
    return '{}b'.format(b)


def clip_cns(leaf_nodes: pd.DataFrame, max_cn: int) -> pd.DataFrame:
    """Clip copy numbers to the range [0, max_cn].

    Parameters
    ----------
    leaf_nodes : pd.DataFrame
        Copy number profiles with a 'CN' column (modified in place).
    max_cn : int
        Maximum copy number.

    Returns
    -------
    pd.DataFrame
        `leaf_nodes`, with 'CN' clipped.
    """
    leaf_nodes['CN'] = np.clip(leaf_nodes['CN'], a_min=0, a_max=max_cn)
    return leaf_nodes


def collapse_identical_consec_bins(nodes: pd.DataFrame,
                                   allele_specific: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge consecutive bins whose copy number is identical in every node into one segment.

    A new segment starts at a bin wherever the copy number differs from the previous bin in at least one node (and,
    if `allele_specific` is False, in at least one allele).

    Parameters
    ----------
    nodes : pd.DataFrame
        Copy number profiles with columns 'Node', 'Chr', 'Pos' (from 1 within each chromosome, or chromosome and
        allele), 'Start', 'End', 'Allele' and 'CN'.
    allele_specific : bool, optional
        If True, the alleles are merged separately, so they can have different breakpoints; if False, the
        breakpoints are shared by both alleles (default: False).

    Returns
    -------
    output : pd.DataFrame
        The merged segments, one row per node, chromosome, segment and allele, with columns 'Node', 'Chr', 'Pos'
        (segment number, from 1 within each chromosome, or chromosome and allele), 'Start', 'End', 'Allele' and 'CN'.
    pos_mapping : pd.DataFrame
        One row per input bin position, chromosome and allele, with columns 'Chr', 'Allele', 'Pos' (the input
        position) and 'group' (its segment number in `output`).
    """
    pos_cn = nodes.pivot_table(index=['Pos'], columns=['Node', 'Chr', 'Allele'], values='CN').fillna('end')
    consec_pos_cn = pos_cn.shift()
    consec_same = pos_cn == consec_pos_cn
    group_cols = ['Chr', 'Allele'] if allele_specific else ['Chr']
    identify_diff_pos = (nodes
                         .merge(pd.melt(consec_same, ignore_index=False).reset_index(),
                                on=['Node', 'Chr', 'Allele', 'Pos'], how='left')
                         .groupby(group_cols + ['Pos'])['value']
                         .apply(lambda x: False in set(x))
                         .reset_index())
    identify_diff_pos['group'] = identify_diff_pos.groupby(group_cols)['value'].cumsum()
    out = nodes.merge(identify_diff_pos[group_cols + ['Pos', 'group']], on=group_cols + ['Pos'])
    output = (out
              .groupby(['Node', 'Chr', 'group', 'Allele'])
              .agg({'Start': 'min', 'End': 'max', 'CN': lambda x: np.unique(x)[0]})
              .reset_index()
              .rename(columns={'group': 'Pos'}))
    pos_mapping = out[['Chr', 'Allele', 'Pos', 'group']].drop_duplicates()
    return output[['Node', 'Chr', 'Pos', 'Start', 'End', 'Allele', 'CN']], pos_mapping


def filter_segments_segsize(filter_nodes: pd.DataFrame, pos_mapping: pd.DataFrame, min_seg_size: float,
                            logger: logging.Logger,
                            keep_consec: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flag segments smaller than `min_seg_size` for the focal evolutionary model, and renumber the rest.

    The largest segment of each chromosome is never flagged. The flags depend only on the segment size and
    chromosome, so both alleles of a segment are flagged alike (the alleles share segments, because the earlier merge
    in `collapse_identical_consec_bins` is not allele specific).

    Parameters
    ----------
    filter_nodes : pd.DataFrame
        Merged segments with columns 'Node', 'Chr', 'Pos', 'Start', 'End', 'Allele', 'CN' and 'seg_size'. A
        'filter_seg' column (and 'consec' if `keep_consec` is True) is added in place.
    pos_mapping : pd.DataFrame
        Position mapping with columns 'Chr', 'Allele', 'origPos' and 'mergePos' (the 'Pos' of `filter_nodes`).
    min_seg_size : float
        Minimum segment size (End - Start) to keep in the ILP.
    logger : logging.Logger
        Logger for the number of segments before and after filtering.
    keep_consec : bool, optional
        If True, small segments next to another small segment (in the same node, chromosome and allele) are kept,
        so only isolated small segments are flagged (default: True).

    Returns
    -------
    filtered_nodes : pd.DataFrame
        The kept segments, with columns 'Node', 'Chr', 'Pos', 'Start', 'End', 'Allele' and 'CN', where 'Pos' is
        renumbered from 1 within each chromosome and allele.
    all_nodes : pd.DataFrame
        All segments, with columns 'Node', 'Chr', 'Pos', 'Start', 'End', 'filter_seg', 'Allele' and 'CN', where
        'filter_seg' is True for segments flagged for the focal evolutionary model.
    pos_mapping : pd.DataFrame
        The input mapping with a 'filteredPos' column added (the new 'Pos', or NaN for flagged segments).
    """
    logger.info('Total number of segments before filtering: {}'
                .format(len(filter_nodes[['Chr', 'Pos', 'Allele']].drop_duplicates())))
    get_largest_chr_seg = (lambda x: x.groupby(['Chr'])['seg_size'].transform('max'))
    filter_nodes['filter_seg'] = np.where((filter_nodes['seg_size'] == get_largest_chr_seg(filter_nodes)) |
                                          (filter_nodes['seg_size'] >= min_seg_size), False, True)
    if keep_consec:
        filter_nodes['consec'] = (filter_nodes
                                  .groupby(['Node', 'Chr', 'Allele'])['filter_seg']
                                  .transform(lambda x: x & ((x == x.shift(1)) | (x == x.shift(-1)))))
        filter_nodes['filter_seg'] = np.where(filter_nodes['consec'], False, filter_nodes['filter_seg'])
        filtered_nodes = (filter_nodes[~filter_nodes.filter_seg]
                          .rename(columns={'Pos': 'mergePos'})
                          .drop(columns=['consec'], axis=1))
    else:
        filtered_nodes = filter_nodes[~filter_nodes.filter_seg].rename(columns={'Pos': 'mergePos'})
    filtered_pos = (filtered_nodes[['Chr', 'Start', 'End', 'mergePos', 'Allele']]
                    .drop_duplicates()
                    .reset_index(drop=True)
                    .sort_values(by=['Chr', 'Allele', 'Start', 'End']))
    filtered_pos['Pos'] = (filtered_pos
                           .groupby(['Chr', 'Allele'])['Start']
                           .transform(lambda x: np.arange(1, x.count() + 1)))
    pos_mapping = (pos_mapping
                   .merge(filtered_pos[['Chr', 'Allele', 'mergePos', 'Pos']],
                          on=['Chr', 'Allele', 'mergePos'], how='left')
                   .rename(columns={'Pos': 'filteredPos'})
                   [['Chr', 'Allele', 'origPos', 'mergePos', 'filteredPos']])
    filtered_nodes = (filtered_nodes
                      .merge(filtered_pos, on=['Chr', 'Allele', 'Start', 'End'], how='left')
                      [['Node', 'Chr', 'Pos', 'Start', 'End', 'Allele', 'CN']])
    logger.info('Total number of segments after filtering: {}'
                .format(len(filtered_nodes[['Chr', 'Pos', 'Allele']].drop_duplicates())))
    return (filtered_nodes, filter_nodes[['Node', 'Chr', 'Pos', 'Start', 'End', 'filter_seg', 'Allele', 'CN']].copy(),
            pos_mapping)


def filter_segments_nsegs(filter_nodes: pd.DataFrame, pos_mapping: pd.DataFrame, n_segs: int,
                          logger: logging.Logger,
                          keep_consec: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep about `n_segs` of the largest segments for the ILP, flag the rest for the focal evolutionary model, and
    renumber the kept segments.

    Segments are ranked per chromosome and allele, so `n_segs` counts each allele of a segment separately. The
    largest segment of each chromosome and allele (the first if tied) is ranked first and always kept; the
    rest are ranked by size. Each fraction in `vals` (0.7, 0.8, 0.9 and 1 if `keep_consec`, otherwise only 1) keeps
    the top round(n_segs * fraction) ranks (`get_best_filter_segs`), and the fraction whose number of kept segments
    is closest to `n_segs` is used. Both alleles of a segment are always filtered together, as the focal
    evolutionary model needs both; so the number kept can be one above `n_segs`.

    Parameters
    ----------
    filter_nodes : pd.DataFrame
        Merged segments with columns 'Node', 'Chr', 'Pos', 'Start', 'End', 'Allele', 'CN' and 'seg_size'. Not
        modified.
    pos_mapping : pd.DataFrame
        Position mapping with columns 'Chr', 'Allele', 'origPos' and 'mergePos' (the 'Pos' of `filter_nodes`).
    n_segs : int
        Target number of segments (per chromosome and allele, summed over the genome) to keep in the ILP.
    logger : logging.Logger
        Logger for the number of segments before and after filtering.
    keep_consec : bool, optional
        If True, some flagged segments are also kept, so that every flagged segment has a neighbour in the ILP (see
        `get_best_filter_segs`) (default: True).

    Returns
    -------
    filtered_nodes : pd.DataFrame
        The kept segments, with columns 'Node', 'Chr', 'Pos', 'Start', 'End', 'Allele' and 'CN', where 'Pos' is
        renumbered from 1 within each chromosome and allele.
    all_nodes : pd.DataFrame
        All segments, with columns 'Node', 'Chr', 'Pos', 'Start', 'End', 'filter_seg', 'Allele' and 'CN', where
        'filter_seg' is True for segments flagged for the focal evolutionary model.
    pos_mapping : pd.DataFrame
        The input mapping with a 'filteredPos' column added (the new 'Pos', or NaN for flagged segments).
    """
    logger.info('Total number of segments before filtering: {}'
                .format(len(filter_nodes[['Chr', 'Pos', 'Allele']].drop_duplicates())))
    leaves_segments_pos = filter_nodes[['Chr', 'Pos', 'Allele', 'Start', 'End', 'seg_size']].drop_duplicates()
    # Mark the largest segment of each chr and allele (the first in input order if tied), which is always kept
    leaves_segments_pos['chr_max_seg'] = ~(leaves_segments_pos
                                           .sort_values(by='seg_size', ascending=False, kind='stable')
                                           .duplicated(subset=['Chr', 'Allele'], keep='first'))
    # Rank the seg sizes where the max seg size for each chr and allele is ranked top
    leaves_segments_pos = (leaves_segments_pos
                           .sort_values(by=['chr_max_seg', 'seg_size'], ascending=False)
                           .reset_index(drop=True)
                           .reset_index()
                           .rename(columns={'index': 'size_rank'}))
    leaves_segments_pos['size_rank'] = leaves_segments_pos['size_rank'] + 1
    vals = [0.7, 0.8, 0.9, 1] if keep_consec else [1]
    filter_segments = min((get_best_filter_segs(leaves_segments_pos, n_segs, prop, keep_consec) for prop in vals),
                          key=lambda x: abs((~x['filter_seg']).sum() - n_segs))
    n_kept = (~filter_segments['filter_seg']).sum()
    if keep_consec and (n_kept != n_segs):
        logger.warning(f'{n_kept} segments are kept in the ILP instead of max_n_segs ({n_segs}), so that every '
                       'flagged segment has a neighbour in the ILP (keep_consec_segs).')
    elif n_kept != n_segs:
        logger.warning(f'{n_kept} segments are kept in the ILP instead of max_n_segs ({n_segs}), so that both '
                       'alleles of each segment and the largest segment of each chromosome are kept.')
    filter_nodes = filter_nodes.merge(filter_segments, on=['Chr', 'Pos', 'Start', 'End', 'Allele'], how='left')
    filtered_nodes = filter_nodes[~filter_nodes.filter_seg].rename(columns={'Pos': 'mergePos'})
    filtered_pos = (filtered_nodes[['Chr', 'Start', 'End', 'mergePos', 'Allele']]
                    .drop_duplicates()
                    .reset_index(drop=True)
                    .sort_values(by=['Chr', 'Allele', 'Start', 'End']))
    filtered_pos['Pos'] = (filtered_pos
                           .groupby(['Chr', 'Allele'])['Start']
                           .transform(lambda x: np.arange(1, x.count() + 1)))
    pos_mapping = (pos_mapping
                   .merge(filtered_pos[['Chr', 'Allele', 'mergePos', 'Pos']],
                          on=['Chr', 'Allele', 'mergePos'], how='left')
                   .rename(columns={'Pos': 'filteredPos'})
                   [['Chr', 'Allele', 'origPos', 'mergePos', 'filteredPos']])
    filtered_nodes = (filtered_nodes
                      .merge(filtered_pos, on=['Chr', 'Allele', 'Start', 'End'], how='left')
                      [['Node', 'Chr', 'Pos', 'Start', 'End', 'Allele', 'CN']])
    logger.info('Total number of segments after filtering: {}'
                .format(len(filtered_nodes[['Chr', 'Pos', 'Allele']].drop_duplicates())))
    return (filtered_nodes, filter_nodes[['Node', 'Chr', 'Pos', 'Start', 'End', 'filter_seg', 'Allele', 'CN']].copy(),
            pos_mapping)


def get_best_filter_segs(leaves_segments: pd.DataFrame, n_segs: int, prop: float, keep_consec: bool) -> pd.DataFrame:
    """Flag the segments outside the top round(n_segs * prop) size ranks, for one fraction tried by
    `filter_segments_nsegs`.

    Segments flagged as 'chr_max_seg' are always kept, and a segment stays flagged only if both its alleles are (so
    both alleles are kept if either is). If `keep_consec` is True, the segments are then split into
    runs of consecutive segments with the same flag, within each chromosome and allele (in genome order), and in
    each run of flagged segments `get_ilp_segs` chooses the fewest segments to keep so that every segment left
    flagged has a neighbour in the ILP.

    Parameters
    ----------
    leaves_segments : pd.DataFrame
        One row per segment, chromosome and allele, with columns 'Chr', 'Pos', 'Allele', 'Start', 'End',
        'seg_size', 'chr_max_seg' and 'size_rank' (from 1). Not modified.
    n_segs : int
        Target number of segments to keep.
    prop : float
        Fraction of `n_segs` to keep by size rank, before any consecutive segments are added back.
    keep_consec : bool
        Whether to also keep segments from runs of consecutive flagged segments.

    Returns
    -------
    pd.DataFrame
        One row per segment, chromosome and allele, sorted by 'Allele', 'Chr', 'Start' and 'End', with columns 'Chr',
        'Pos', 'Allele', 'Start', 'End', 'seg_size' and 'filter_seg' (True if flagged).
    """
    leaves_segments_pos = leaves_segments.copy()
    # Select the top n_segs segments and the segments that are the max in the chr to keep, the rest are set to filter.
    leaves_segments_pos['filter_seg'] = np.where(leaves_segments_pos['size_rank'] <= round(n_segs * prop),
                                                 False,
                                                 ~leaves_segments_pos['chr_max_seg'])
    # Keep both alleles of a segment in the ILP if either is kept (the focal evolutionary model needs both alleles)
    leaves_segments_pos['filter_seg'] = (leaves_segments_pos
                                         .groupby(['Chr', 'Pos'])['filter_seg']
                                         .transform('all'))
    leaves_segments_pos = leaves_segments_pos.sort_values(by=['Allele', 'Chr', 'Start', 'End'])
    if keep_consec:
        # Get groups of consecutive segments that have the same filter status, within each chromosome and allele
        run_cols = leaves_segments_pos[['Chr', 'Allele', 'filter_seg']]
        leaves_segments_pos['group'] = (run_cols != run_cols.shift(1)).any(axis=1).cumsum()
        chr_cols = leaves_segments_pos[['Chr', 'Allele']]
        leaves_segments_pos['chr_start'] = (chr_cols != chr_cols.shift(1)).any(axis=1)
        leaves_segments_pos['chr_end'] = (chr_cols != chr_cols.shift(-1)).any(axis=1)
        # Move the fewest flagged segments into the ILP so that every flagged segment has a neighbour in the ILP
        to_ilp = (leaves_segments_pos
                  .groupby('group', group_keys=False)[['seg_size', 'chr_start', 'chr_end']]
                  .apply(lambda run: pd.Series(get_ilp_segs(run['seg_size'].to_numpy(), run['chr_start'].iloc[0],
                                                            run['chr_end'].iloc[-1]),
                                               index=run.index)))
        leaves_segments_pos['filter_seg'] = leaves_segments_pos['filter_seg'] & ~to_ilp
    return leaves_segments_pos[['Chr', 'Pos', 'Allele', 'Start', 'End', 'seg_size', 'filter_seg']].copy()


def get_ilp_segs(sizes: np.ndarray, chr_start: bool, chr_end: bool) -> np.ndarray:
    """Choose which segments of a run of consecutive flagged segments move into the ILP, so that every segment
    left flagged has a neighbour in the ILP.

    Every third segment moves, from the offset (0, 1 or 2) that moves the fewest segments, then the largest total
    size. The segments either side of the run are in the ILP, unless the run starts or ends its chromosome.

    Parameters
    ----------
    sizes : np.ndarray
        Sizes of the segments in the run, in genome order.
    chr_start : bool
        Whether the run starts its chromosome (no segment before it).
    chr_end : bool
        Whether the run ends its chromosome (no segment after it).

    Returns
    -------
    np.ndarray
        Boolean array, True for the segments that move into the ILP.
    """
    best = None
    for offset in range(3):
        to_ilp = np.zeros(len(sizes), dtype=bool)
        to_ilp[offset::3] = True
        in_ilp = np.concatenate([[not chr_start], to_ilp, [not chr_end]])
        if not (to_ilp | in_ilp[:-2] | in_ilp[2:]).all():
            continue
        score = (to_ilp.sum(), -sizes[to_ilp].sum())
        if (best is None) or (score < best[0]):
            best = (score, to_ilp)
    return best[1]


def convert_node_names(leaves: np.ndarray, fixed_topology: Union[bool, Sequence[tuple], str] = False) -> dict:
    """Map node names to the integer node IDs used in the ILP: internal nodes first, then the leaves.

    If the topology is not fixed, IDs 0 to n_leaves - 1 are kept for the internal nodes (mapped to themselves) and
    the leaves get n_leaves to 2 * n_leaves - 1, in sorted order of their names (numbers inside names compared as
    numbers, so 'clone_2' comes before 'clone_10'), so the IDs don't depend on the order of the input rows. If it is
    fixed, the internal nodes keep their names if these are integers and the root is 0; otherwise they are renumbered
    from the root (0) by `rename_internal_nodes`. The leaves are then numbered from one above the largest internal
    ID, in the same sorted order.

    Parameters
    ----------
    leaves : np.ndarray
        Leaf names, in any order.
    fixed_topology : bool, sequence of tuple or str, optional
        False if the topology is inferred (default); otherwise the tree edges as (parent, child) node names, or a
        string of them, which is evaluated.

    Returns
    -------
    dict
        Integer node ID for each node name (and for each internal ID, if the topology is not fixed).
    """
    # Number the leaves in sorted order, so their IDs don't depend on the order of the input rows
    leaves = sorted(leaves, key=lambda x: tuple(int(p) if p.isdigit() else p for p in re.split(r'(\d+)', str(x))))
    n_leaves = len(leaves)
    if not fixed_topology:
        internal_names = list(range(0, n_leaves))
        leaf_mapping = dict(zip(leaves, range(n_leaves, 2 * n_leaves)))
        name_mapping = {i: i for i in internal_names} | leaf_mapping
    else:
        fixed_topology = eval(fixed_topology) if isinstance(fixed_topology, str) else fixed_topology
        assert all(len(e) == 2 for e in fixed_topology), 'Each edge of fixed_topology must be a (parent, child) tuple'
        parents, children = list(zip(*fixed_topology))
        parents = list(set(parents))
        roots = [i for i in parents if i not in children]
        assert len(roots) == 1, f'fixed_topology must have exactly one root, found {len(roots)}: {roots}'
        root = roots[0]
        if any(type(i) != int for i in parents) or (root != 0):
            internal_mapping, _ = rename_internal_nodes(root, parents, fixed_topology, 0, {})
        else:
            internal_mapping = {i: i for i in parents}
        leaf_start = max(list(internal_mapping.values())) + 1
        leaf_mapping = dict(zip(leaves, range(leaf_start, leaf_start + n_leaves)))
        name_mapping = internal_mapping | leaf_mapping
    return name_mapping


def rename_internal_nodes(node: Union[int, str], internal_nodes: list, edges: Sequence[tuple], count: int = 0,
                          mapping: dict = {}) -> tuple[dict, int]:
    """Number `node` and the internal nodes below it depth-first (pre-order), starting from `count`.

    Children are visited in the order of `edges`. Called by `convert_node_names` from the root with count 0.

    Parameters
    ----------
    node : int or str
        Node to number next.
    internal_nodes : list
        Names of the internal nodes (parents in `edges`); leaves are not numbered.
    edges : sequence of tuple
        Tree edges as (parent, child) node names.
    count : int, optional
        ID to give `node` (default: 0).
    mapping : dict, optional
        IDs assigned so far, {node name: ID}; modified in place. The default dict is shared between calls, so
        callers should pass a new dict (default: {}).

    Returns
    -------
    mapping : dict
        `mapping` with `node` and the internal nodes below it added.
    count : int
        The next unused ID.
    """
    mapping[node] = count
    count += 1
    node_children = [j for i, j in edges if (i == node) and (j in internal_nodes)]
    if (len(node_children) == 0) or (all(i in list(mapping.keys()) for i in node_children)):
        return mapping, count
    else:
        for child in node_children:
            mapping, count = rename_internal_nodes(child, internal_nodes, edges, count, mapping)
        return mapping, count
