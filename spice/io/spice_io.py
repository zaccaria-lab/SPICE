import pandas as pd
import numpy as np

from spice.io.phasecns import phasing
from spice.io.logging import *

def process_spice_input(leaves_df, max_cn, phase_alleles, allele_specific, leaf_dependent_phasing, min_seg_size, 
                        max_n_segs, keep_consec_segs, focal_max_cn, fixed_topology, logger=None):
    
    leaves = leaves_df.copy()
    if logger is None:
        logger, logfile = initialize_logger(True, None)
    # Process allele specific input
    if allele_specific == False:
        if 'Allele' in leaves.columns:
            if leaves.Allele.nunique() > 1:
                logger.warning('Allele specific mode is turned off, but input data contains multiple alleles. \
                Condensing to total copy numbers.')
                leaves = (leaves
                         .groupby(['Node', 'Chr', 'Pos', 'Start', 'End'])
                         .agg({'CN' : 'sum'})
                         .reset_index()
                         .assign(Allele='A'))
        else:
            # If total copy number we just assign all to 'A' to match ILP input format.
            leaves['Allele'] = 'A'     
    # Phase alleles 
    if phase_alleles:
        if allele_specific:
            logger.info(f"Phasing alleles : {'leaf_dependent' if leaf_dependent_phasing else 'leaf_independent'}")
            leaves = phase_profiles(leaves, leaf_dependent=leaf_dependent_phasing)
        else:
            logger.warning(f"Phasing alleles not supported for non-allele specific mode. Ignoring phasing parameter.")
    # Collapse CN bins to segments
    logger.trace(f'Collapsing CN bins to segments.')
    original_genome_locs = (leaves[['Chr', 'Allele', 'Start', 'End', 'Pos']]
                            .drop_duplicates() 
                            .sort_values(by=['Chr', 'Allele', 'Start', 'End']) 
                            .rename(columns={'Pos' : 'origPos'}))
    leaves_segments, pos_mapping = collapse_identical_consec_bins(leaves, allele_specific=False)
    pos_mapping = pos_mapping.rename(columns={'Pos' : 'origPos',
                                              'group' : 'mergePos'})
    # Separate small and large segments (based on min_seg_size) 
    leaves_segments['seg_size'] = leaves_segments['End'] - leaves_segments['Start']
    consec_msg = ', plus any consecutive segments' if keep_consec_segs else ''
    if max_n_segs is not None:
        if min_seg_size > 0:
            logger.warning(f'Both max_n_segs and min_seg_size are specified. Filtering segments based on max_n_segs : \
the largest {max_n_segs} segments will be selected{consec_msg}.')
        else:
            logger.trace(f'Selecting the largest {max_n_segs} segments{consec_msg}.')
        filtered_leaves, removed_seg_nodes, filt_pos_mapping = filter_segments_nsegs(leaves_segments, pos_mapping, 
                                                                                     n_segs=max_n_segs, logger=logger, 
                                                                                     keep_consec=keep_consec_segs) 
    else:   
        if min_seg_size > 0:
            logger.trace(f'Selecting segments above a min_seg_size of {base_converter(min_seg_size)}{consec_msg}.')
        else:
            logger.trace(f'No min_seg_size or maximum number of segments specified. All segments will be kept.')
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
        logger.trace(f'Clipping interval model CN profiles to maximum value of {max_cn}. Previous maximum value was \
{current_max_cn}. {prop_above_threshold:.2%} of input CN values were above the threshold.')
        spice_input_nodes = clip_cns(spice_input_nodes, max_cn=max_cn)

        focal_max_cn = focal_max_cn if focal_max_cn is not None else max_cn 
        current_focal_max_cn = removed_seg_nodes.CN.max()
        prop_above_threshold = (removed_seg_nodes.CN > max_cn).sum() / len(removed_seg_nodes)
        logger.trace(f'Clipping small segment model CN profiles to maximum value of {focal_max_cn}. Previous maximum \
value was {current_focal_max_cn}. {prop_above_threshold:.2%} of input CN values were above the threshold.')
        removed_seg_nodes = clip_cns(removed_seg_nodes, max_cn=focal_max_cn)
    # Process mapping of original positions to final positions   
    final_mapping = final_mapping.rename(columns={'Pos' : 'filteredPos',
                                                  'group' : 'finalPos'})
    final_mapping = original_genome_locs.merge(filt_pos_mapping, on=['Chr', 'Allele', 'origPos'], how='left') \
                                        .merge(final_mapping, on=['Chr', 'Allele', 'filteredPos'], how='left') \
                                        .astype(int, errors='ignore')
    # Convert bootstrap leaves input to dictionary (bootstrap leaves are not filtered to remove small segments).
    node_name_mapping = convert_node_names(leaves['Node'].unique(), fixed_topology)
    leaves_segments_dict = cn_df_to_dict(leaves_segments, node_name_mapping, allele_specific)
    boot_segsizes = (leaves_segments[['Chr', 'Allele', 'Pos', 'seg_size']]
                     .drop_duplicates().sort_values(by=['Chr', 'Pos', 'Allele'])
                     .groupby(['Chr', 'Allele'])['seg_size']
                     .apply(list)
                     .reset_index()
                     .groupby('Chr')['seg_size']
                     .apply(list)
                     .to_dict())
    # Convert ILP leaves input to dictionary
    spice_input_dict = cn_df_to_dict(spice_input_nodes, node_name_mapping, allele_specific)
    test_node = list(spice_input_dict.keys())[0]
    logger.info(f"Processed input CN profiles. {len(spice_input_dict)} leaves with \
{len(final_mapping[['Chr', 'Allele', 'finalPos']].drop_duplicates())} final segments")
    if allele_specific:
        chr_lengths = [f'{ch} : [A: {len(spice_input_dict[test_node][ch][0])} B: {len(spice_input_dict[test_node][ch][1])}]' 
                       for ch in spice_input_dict[test_node]]
    else:
        chr_lengths = [f'{ch} : {len(spice_input_dict[test_node][ch][0])}' 
                       for ch in spice_input_dict[test_node]]
    logger.info(f"Chromosome lengths (ILP input): ({', '.join(chr_lengths)})")
    return leaves_segments_dict, boot_segsizes, spice_input_dict, node_name_mapping, removed_seg_nodes, final_mapping

base_converter = (lambda b: '{}mb'.format(b / 1000000) if b >= 1000000 else '{}kb'.format(b / 1000) if b >= 1000 else '{}b'.format(b))

def clip_cns(leaf_nodes, max_cn):
    leaf_nodes['CN'] = np.clip(leaf_nodes['CN'], a_min=0, a_max=max_cn)
    return leaf_nodes


def collapse_identical_consec_bins(nodes, allele_specific=False):
    if 'Pos' not in nodes.columns:
        nodes['Pos'] = (nodes
                        .sort_values(by=['Node', 'Chr', 'Start', 'End', 'Allele'])
                        .groupby(['Node', 'Chr', 'Allele'])['Start']
                        .transform(lambda x: np.arange(len(x))))
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
    out = nodes.merge(identify_diff_pos[group_cols + ['Pos', 'group']], on= group_cols + ['Pos'])
    output = (out
              .groupby(['Node', 'Chr', 'group', 'Allele'])
              .agg({'Start' : 'min', 'End' : 'max', 'CN' : (lambda x: np.unique(x)[0])})
              .reset_index()
              .rename(columns={'group' : 'Pos'}))
    pos_mapping = out[['Chr', 'Allele', 'Pos', 'group']].drop_duplicates()
    return output[['Node', 'Chr', 'Pos', 'Start', 'End', 'Allele', 'CN']], pos_mapping


# Filter segments can filter different segments for each allele, however input merging is not allele specific so the 
# same segments are merged for both alleles - can edit previous functions in future
def filter_segments_segsize(filter_nodes, pos_mapping, min_seg_size, logger, keep_consec=True):
    logger.info('Total number of segments before filtering: {}'.format(len(filter_nodes[['Chr', 'Pos', 'Allele']].drop_duplicates())))
    get_largest_chr_seg = (lambda x: filter_nodes.groupby(['Chr'])['seg_size'].transform('max'))
    filter_nodes['filter_seg'] = np.where((filter_nodes['seg_size'] == get_largest_chr_seg(filter_nodes)) | 
                                          (filter_nodes['seg_size'] >= min_seg_size), False, True)
    if keep_consec:
        filter_nodes['consec'] = (filter_nodes
                                  .groupby(['Node', 'Chr', 'Allele'])['filter_seg']
                                  .transform((lambda x: (x) & ((x == x.shift(1)) | (x == x.shift(-1))))))
        filter_nodes['filter_seg'] = np.where(filter_nodes['consec'], False, filter_nodes['filter_seg'])
        filtered_nodes = (filter_nodes[~filter_nodes.filter_seg]
                          .rename(columns={'Pos' : 'mergePos'})
                          .drop(columns=['consec'], axis=1))
    else:
        filtered_nodes = filter_nodes[~filter_nodes.filter_seg].rename(columns={'Pos' : 'mergePos'})
    filtered_pos = (filtered_nodes[['Chr', 'Start', 'End', 'mergePos', 'Allele']]
                    .drop_duplicates()
                    .reset_index(drop=True)
                    .sort_values(by=['Chr', 'Allele', 'Start', 'End']))
    filtered_pos['Pos'] = filtered_pos.groupby(['Chr', 'Allele'])['Start'].transform((lambda x: np.arange(1, x.count() + 1)))
    pos_mapping = (pos_mapping
                   .merge(filtered_pos[['Chr', 'Allele', 'mergePos', 'Pos']], on=['Chr', 'Allele', 'mergePos'], how='left')
                   .rename(columns={'Pos' : 'filteredPos'})[['Chr', 'Allele', 'origPos', 'mergePos', 'filteredPos']])
    filtered_nodes = filtered_nodes.merge(filtered_pos, on=['Chr', 'Allele', 'Start', 'End'], 
                                          how='left')[['Node', 'Chr', 'Pos', 'Start', 'End', 'Allele', 'CN']]
    logger.info('Total number of segments after filtering: {}'.format(len(filtered_nodes[['Chr', 'Pos', 'Allele']].drop_duplicates())))
    return filtered_nodes, filter_nodes[['Node', 'Chr', 'Pos', 'Start', 'End', 'filter_seg', 'Allele', 'CN']].copy(), pos_mapping

def filter_segments_nsegs(filter_nodes, pos_mapping, n_segs, logger, keep_consec=True):
    logger.info('Total number of segments before filtering: {}'.format(len(filter_nodes[['Chr', 'Pos', 'Allele']].drop_duplicates())))
    leaves_segments_pos = filter_nodes[['Chr', 'Pos', 'Allele', 'Start', 'End', 'seg_size']].drop_duplicates()
    # Get the max seg size for each chr and allele
    leaves_segments_pos['chr_max_seg'] = ~leaves_segments_pos.duplicated(subset=['Chr', 'Allele'], keep='first') 
    # Rank the seg sizes where the max seg size for each chr and allele is ranked top
    leaves_segments_pos = (leaves_segments_pos
                           .sort_values(by=['chr_max_seg', 'seg_size'], ascending=False)
                           .reset_index(drop=True)
                           .reset_index()
                           .rename(columns={'index' : 'size_rank'}))
    leaves_segments_pos['size_rank'] = leaves_segments_pos['size_rank'] + 1
    vals = [0.7, 0.8, 0.9, 1] if keep_consec else [1]
    filter_segments = min((get_best_filter_segs(leaves_segments_pos, n_segs, prop, keep_consec) for prop in vals), 
                          key=(lambda x: abs((~x['filter_seg']).sum() - n_segs)))
    filter_nodes = filter_nodes.merge(filter_segments, on=['Chr', 'Pos', 'Start', 'End', 'Allele'], how='left')
    filtered_nodes = filter_nodes[~filter_nodes.filter_seg].rename(columns={'Pos' : 'mergePos'})
    filtered_pos = (filtered_nodes[['Chr', 'Start', 'End', 'mergePos', 'Allele']]
                    .drop_duplicates()
                    .reset_index(drop=True)
                    .sort_values(by=['Chr', 'Allele', 'Start', 'End']))
    filtered_pos['Pos'] = filtered_pos.groupby(['Chr', 'Allele'])['Start'].transform((lambda x: np.arange(1, x.count() + 1)))
    pos_mapping = (pos_mapping
                   .merge(filtered_pos[['Chr', 'Allele', 'mergePos', 'Pos']], on=['Chr', 'Allele', 'mergePos'], how='left')
                   .rename(columns={'Pos' : 'filteredPos'})
                   [['Chr', 'Allele', 'origPos', 'mergePos', 'filteredPos']])
    filtered_nodes = (filtered_nodes
                      .merge(filtered_pos, on=['Chr', 'Allele', 'Start', 'End'], how='left')
                      [['Node', 'Chr', 'Pos', 'Start', 'End', 'Allele', 'CN']])
    logger.info('Total number of segments after filtering: {}'.format(len(filtered_nodes[['Chr', 'Pos', 'Allele']].drop_duplicates())))
    return filtered_nodes, filter_nodes[['Node', 'Chr', 'Pos', 'Start', 'End', 'filter_seg', 'Allele', 'CN']].copy(), pos_mapping
    
def get_best_filter_segs(leaves_segments, n_segs, prop, keep_consec):
    leaves_segments_pos = leaves_segments.copy()
    # Select the top n_segs segments and the segments that are the max in the chr to keep, the rest are set to filter.
    leaves_segments_pos['filter_seg'] = np.where(leaves_segments_pos['size_rank'] <= round(n_segs * prop), 
                                                 False, 
                                                 ~leaves_segments_pos['chr_max_seg'])
    leaves_segments_pos = leaves_segments_pos.sort_values(by=['Allele', 'Chr', 'Start', 'End'])
    if keep_consec:
        # Get groups of consecutive segments that have the same filter status
        leaves_segments_pos['group'] = (leaves_segments_pos['filter_seg'] != leaves_segments_pos['filter_seg'].shift(1)).cumsum()
        # Mark the segments from each consecutive group to keep (that were not already being kept) - this is done by 
        # selecting the fewest segments possible such that 3 consecutive segments are not filtered, where ties are broken 
        # by selecting the largest segments.
        leaves_segments_pos['filter_seg'] = np.where(~leaves_segments_pos['filter_seg'], 
                                                     False, 
                                                     (leaves_segments_pos
                                                      .groupby('group')['seg_size']
                                                      .transform(lambda x: get_keep_idx(x))))
    return leaves_segments_pos[['Chr', 'Pos', 'Allele', 'Start', 'End', 'seg_size', 'filter_seg']].copy()

def get_keep_idx(x):
    keep = np.zeros(len(x))
    keep[get_min_idx(x)::3] = 1
    return keep.astype(bool)

get_min_idx = (lambda x: min(((i, x[i::3]) for i in range(3)),  key=(lambda p: len(p[1]) + (1 - (sum(p[1]) / sum(x)))))[0])

def cn_dict_to_df(cn_dict):
    cn_df = pd.DataFrame([{'Node': node, 'Chr' : ch, 'Allele' : ['A', 'B'][allele], 'Pos' : pos+1, 'CN' : cn} 
                          for node in cn_dict 
                          for ch in cn_dict[node] 
                          for allele in range(len(cn_dict[node][ch])) 
                          for pos, cn in enumerate(cn_dict[node][ch][allele])])
    return cn_df


def cn_df_to_dict(cn_df, node_conversion=None, allele_specific=True):
    if node_conversion is None:
        cn_dict = {node : {ch : {allele : cn_df[(cn_df.Node == node) & 
                                                (cn_df.Chr == ch) & 
                                                (cn_df.Allele == ['A', 'B'][allele])].CN.values.tolist() 
                                 for allele in range(1+allele_specific)} 
                           for ch in cn_df.Chr.unique()} 
                   for node in cn_df.Node.unique()}
    else:
        cn_dict = {node_conversion[node] : {ch : {allele : cn_df[(cn_df.Node == node) & 
                                                                 (cn_df.Chr == ch) & 
                                                                 (cn_df.Allele == ['A', 'B'][allele])].CN.values.tolist() 
                                                  for allele in range(1+allele_specific)} 
                                            for ch in cn_df.Chr.unique()} 
                                        for node in cn_df.Node.unique()}
    return cn_dict
    
def convert_node_names(leaves, fixed_topology=False):
    n_leaves = len(leaves)
    if fixed_topology == False:
        internal_names = list(range(0, n_leaves))
        leaf_mapping = dict(zip(leaves, range(n_leaves, 2*n_leaves)))
        name_mapping = {i : i for i in internal_names} | leaf_mapping
    else:
        fixed_topology = eval(fixed_topology) if type(fixed_topology) == str else fixed_topology
        parents, children = list(zip(*fixed_topology))
        parents = list(set(parents))
        root = [i for i in parents if i not in children][0]
        if all(type(i) != int for i in parents) or (root != 0):
            internal_mapping, _ = rename_internal_nodes(root, parents, fixed_topology, 0, {})
        else:
            internal_mapping = {i : i for i in parents}
        leaf_start = max(list(internal_mapping.values())) + 1
        leaf_mapping = dict(zip(leaves, range(leaf_start, leaf_start + n_leaves)))
        name_mapping = internal_mapping | leaf_mapping
    return name_mapping

def rename_internal_nodes(node, internal_nodes, edges, count=0, mapping={}):
    mapping[node] = count
    count += 1
    node_children = [j for i, j in edges if (i == node) & (j in internal_nodes)]
    if (len(node_children) == 0) or (all(i in list(mapping.keys()) for i in node_children)):
        return mapping, count
    else:
        for child in node_children:
            mapping, count = rename_internal_nodes(child, internal_nodes, edges, count, mapping)
        return mapping, count
    
    
    
    

    