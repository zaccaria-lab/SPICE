import pandas as pd
import numpy as np

from spice.io.phasecns import phasing
from spice.io.logging import *

def process_spice_input(leaves_df, max_cn, phase_alleles, allele_specific, leaf_dependent_phasing, min_seg_size, max_n_segs, keep_consec_segs, logger=None):
    
    leaves = leaves_df.copy()
    
    if (allele_specific == False) & (leaves.Allele.nunique() > 1):
        logger.warning('Allele specific mode is turned off, but input data contains multiple alleles. Condensing to total copy numbers.')
        leaves = leaves.groupby(['Node', 'Chr', 'Pos', 'Start', 'End']).agg({'CN' : 'sum'}).reset_index().assign(Allele='A')
    if logger is None:
        logger, logfile = initialize_logger(True, None)
    # Phase alleles 
    if phase_alleles:
        if allele_specific:
            logger.info(f"Phasing alleles : {'leaf_dependent' if leaf_dependent_phasing else 'leaf_independent'}")
            leaves = phase_profiles(leaves, leaf_dependent=leaf_dependent_phasing)
        else:
            logger.warning(f"Phasing alleles not supported for non-allele specific mode. Ignoring phasing parameter.")
    # Collapse CN bins to segments
    logger.trace(f'Collapsing CN bins to segments.')
    original_genome_locs = leaves[['Chr', 'Allele', 'Start', 'End', 'Pos']].drop_duplicates() \
                                                                           .sort_values(by=['Chr', 'Allele', 'Start', 'End']) \
                                                                           .rename(columns={'Pos' : 'origPos'})
    leaves_segments, pos_mapping = collapse_identical_consec_bins(leaves, allele_specific=False)
    pos_mapping = pos_mapping.rename(columns={'Pos' : 'origPos',
                                              'group' : 'mergePos'})
    # Separate small and large segments (based on min_seg_size) #! FUTURE: CHNAGE THIS SO THAT FILTER SEGS IS ALLELE SPECIFIC!!
    leaves_segments['seg_size'] = leaves_segments['End'] - leaves_segments['Start']
    consec_msg = ', plus any consecutive segments' if keep_consec_segs else ''
    if max_n_segs is not None:
        if min_seg_size > 0:
            logger.warning(f'Both max_n_segs and min_seg_size are specified. Filtering segments based on max_n_segs : the largest {max_n_segs} segments will be selected{consec_msg}.')
        else:
            logger.trace(f'Selecting the largest {max_n_segs} segments{consec_msg}.')
        filtered_leaves, removed_seg_nodes, filt_pos_mapping = filter_segments_nsegs(leaves_segments, pos_mapping, n_segs=max_n_segs, logger=logger, keep_consec=keep_consec_segs) 
    else:   
        if min_seg_size > 0:
            logger.trace(f'Selecting segments above a min_seg_size of {base_converter(min_seg_size)}{consec_msg}.')
        else:
            logger.trace(f'No min_seg_size or maximum number of segments specified. All segments will be kept.')
        filtered_leaves, removed_seg_nodes, filt_pos_mapping = filter_segments_segsize(leaves_segments, pos_mapping, min_seg_size=min_seg_size, logger=logger, keep_consec=keep_consec_segs)
    
    # Rejoin large segments
    spice_input_nodes, final_mapping = collapse_identical_consec_bins(filtered_leaves, allele_specific=True)
    # Clip CNs of spice_input to max_cn
    if max_cn is not None:
        current_max_cn = spice_input_nodes.CN.max()
        prop_above_threshold = (spice_input_nodes.CN > max_cn).sum() / len(spice_input_nodes)
        logger.trace(f'Clipping CN profiles to maximum value of {max_cn}. Previous maximum value was {current_max_cn}. ' \
                     f'{prop_above_threshold:.2%} of input CN values were above the threshold.')
        spice_input_nodes = clip_cns(spice_input_nodes, max_cn=max_cn)
    # Process mapping of original positions to final positions   
    final_mapping = final_mapping.rename(columns={'Pos' : 'filteredPos',
                                                  'group' : 'finalPos'})
    final_mapping = original_genome_locs.merge(filt_pos_mapping, on=['Chr', 'Allele', 'origPos'], how='left') \
                                        .merge(final_mapping, on=['Chr', 'Allele', 'filteredPos'], how='left') \
                                        .astype(int, errors='ignore')
    # Convert bootstrap leaves input to dictionary (bootstrap leaves are not filtered to remove small segments).
    leaves_segments_dict, leaves_segments_mapping = cn_df_to_dict(leaves_segments, True, allele_specific)
    boot_segsizes = leaves_segments[['Chr', 'Allele', 'Pos', 'seg_size']].drop_duplicates().sort_values(by=['Chr', 'Pos', 'Allele']) \
                                                                         .groupby(['Chr', 'Allele'])['seg_size'].apply(list).reset_index() \
                                                                         .groupby('Chr')['seg_size'].apply(list).to_dict()
    # Convert ILP leaves input to dictionary
    spice_input_dict, spice_input_mapping = cn_df_to_dict(spice_input_nodes, True, allele_specific)
    
    logger.info(f"Processed input CN profiles. {len(spice_input_dict)} leaves with {len(final_mapping[['Chr', 'Allele', 'finalPos']].drop_duplicates())} final segments")
    if allele_specific:
        logger.info(f"Chromosome lengths (ILP input): ({', '.join([f'{ch} : [A: {len(spice_input_dict[1][ch][0])} B: {len(spice_input_dict[1][ch][1])}]' for ch in spice_input_dict[1]])})")
    else:
        logger.info(f"Chromosome lengths (ILP input): ({', '.join([f'{ch} : [Total alleles: {len(spice_input_dict[1][ch][0])}]' for ch in spice_input_dict[1]])})")

    return leaves_segments_dict, boot_segsizes, spice_input_dict, spice_input_mapping, removed_seg_nodes, final_mapping

base_converter = (lambda b: '{}mb'.format(b / 1000000) if b >= 1000000 else '{}kb'.format(b / 1000) if b >= 1000 else '{}b'.format(b))

def clip_cns(leaf_nodes, max_cn):
    leaf_nodes['CN'] = np.clip(leaf_nodes['CN'], a_min=0, a_max=max_cn)
    return leaf_nodes


def collapse_identical_consec_bins(nodes, allele_specific=False):
    if 'Pos' not in nodes.columns:
        nodes['Pos'] = nodes.sort_values(by=['Node', 'Chr', 'Start', 'End', 'Allele']).groupby(['Node', 'Chr', 'Allele'])['Start'].transform(lambda x: np.arange(len(x)))
    pos_cn = nodes.pivot_table(index=['Pos'], columns=['Node', 'Chr', 'Allele'], values='CN').fillna('end')
    consec_pos_cn = pos_cn.shift()
    consec_same = pos_cn == consec_pos_cn
    group_cols = ['Chr', 'Allele'] if allele_specific else ['Chr']
    identify_diff_pos = nodes.merge(pd.melt(consec_same, ignore_index=False).reset_index(), on=['Node', 'Chr', 'Allele', 'Pos'], how='left').groupby(group_cols + ['Pos'])['value'].apply(lambda x: False in set(x)).reset_index()
    identify_diff_pos['group'] = identify_diff_pos.groupby(group_cols)['value'].cumsum()
    out = nodes.merge(identify_diff_pos[group_cols + ['Pos', 'group']], on= group_cols + ['Pos'])
    output = out.groupby(['Node', 'Chr', 'group', 'Allele']).agg({'Start' : 'min', 'End' : 'max', 'CN' : (lambda x: np.unique(x)[0])}).reset_index().rename(columns={'group' : 'Pos'})
    pos_mapping = out[['Chr', 'Allele', 'Pos', 'group']].drop_duplicates()
    return output[['Node', 'Chr', 'Pos', 'Start', 'End', 'Allele', 'CN']], pos_mapping


# Filter segments can filter different segments for each allele, however input merging is not allele specific so the same segments are merged for both alleles - can edit previous functions in future
def filter_segments_segsize(filter_nodes, pos_mapping, min_seg_size, logger, keep_consec=True):
    logger.info('Total number of segments before filtering: {}'.format(len(filter_nodes[['Chr', 'Pos', 'Allele']].drop_duplicates())))
    filter_nodes['filter_seg'] = np.where((filter_nodes['seg_size'] == filter_nodes.groupby(['Chr'])['seg_size'].transform('max')) | (filter_nodes['seg_size'] >= min_seg_size), False, True)
    if keep_consec:
        filter_nodes['consec'] = filter_nodes.groupby(['Node', 'Chr', 'Allele'])['filter_seg'].transform((lambda x: (x) & ((x == x.shift(1)) | (x == x.shift(-1))))) 
        filter_nodes['filter_seg'] = np.where(filter_nodes['consec'], False, filter_nodes['filter_seg'])
        filtered_nodes = filter_nodes[~filter_nodes.filter_seg].rename(columns={'Pos' : 'mergePos'}).drop(columns=['consec'], axis=1)
    else:
        filtered_nodes = filter_nodes[~filter_nodes.filter_seg].rename(columns={'Pos' : 'mergePos'})
    filtered_pos = filtered_nodes[['Chr', 'Start', 'End', 'mergePos', 'Allele']].drop_duplicates().reset_index(drop=True).sort_values(by=['Chr', 'Allele', 'Start', 'End'])
    filtered_pos['Pos'] = filtered_pos.groupby(['Chr', 'Allele'])['Start'].transform((lambda x: np.arange(1, x.count() + 1)))
    pos_mapping = pos_mapping.merge(filtered_pos[['Chr', 'Allele', 'mergePos', 'Pos']], on=['Chr', 'Allele', 'mergePos'], how='left').rename(columns={'Pos' : 'filteredPos'})[['Chr', 'Allele', 'origPos', 'mergePos', 'filteredPos']]
    filtered_nodes = filtered_nodes.merge(filtered_pos, on=['Chr', 'Allele', 'Start', 'End'], how='left')[['Node', 'Chr', 'Pos', 'Start', 'End', 'Allele', 'CN']]
    logger.info('Total number of segments after filtering: {}'.format(len(filtered_nodes[['Chr', 'Pos', 'Allele']].drop_duplicates())))
    return filtered_nodes, filter_nodes[['Node', 'Chr', 'Pos', 'Start', 'End', 'filter_seg', 'Allele', 'CN']], pos_mapping

def filter_segments_nsegs(filter_nodes, pos_mapping, n_segs, logger, keep_consec=True):
    logger.info('Total number of segments before filtering: {}'.format(len(filter_nodes[['Chr', 'Pos', 'Allele']].drop_duplicates())))
    leaves_segments_pos = filter_nodes[['Chr', 'Pos', 'Allele', 'Start', 'End', 'seg_size']].drop_duplicates()
    # Get the max seg size for each chr and allele
    leaves_segments_pos['chr_max_seg'] = ~leaves_segments_pos.duplicated(subset=['Chr', 'Allele'], keep='first') 
    # Rank the seg sizes where the max seg size for each chr and allele is ranked top
    leaves_segments_pos = leaves_segments_pos.sort_values(by=['chr_max_seg', 'seg_size'], ascending=False).reset_index(drop=True).reset_index().rename(columns={'index' : 'size_rank'}) 
    leaves_segments_pos['size_rank'] = leaves_segments_pos['size_rank'] + 1
    vals = [0.7, 0.8, 0.9, 1] if keep_consec else [1]
    filter_segments = min((get_best_filter_segs(leaves_segments_pos, n_segs, prop, keep_consec) for prop in vals), key=(lambda x: abs((~x['filter_seg']).sum() - n_segs)))
    filter_nodes = filter_nodes.merge(filter_segments, on=['Chr', 'Pos', 'Start', 'End', 'Allele'], how='left')
    filtered_nodes = filter_nodes[~filter_nodes.filter_seg].rename(columns={'Pos' : 'mergePos'})
    filtered_pos = filtered_nodes[['Chr', 'Start', 'End', 'mergePos', 'Allele']].drop_duplicates().reset_index(drop=True).sort_values(by=['Chr', 'Allele', 'Start', 'End'])
    filtered_pos['Pos'] = filtered_pos.groupby(['Chr', 'Allele'])['Start'].transform((lambda x: np.arange(1, x.count() + 1)))
    pos_mapping = pos_mapping.merge(filtered_pos[['Chr', 'Allele', 'mergePos', 'Pos']], on=['Chr', 'Allele', 'mergePos'], how='left').rename(columns={'Pos' : 'filteredPos'})[['Chr', 'Allele', 'origPos', 'mergePos', 'filteredPos']]
    filtered_nodes = filtered_nodes.merge(filtered_pos, on=['Chr', 'Allele', 'Start', 'End'], how='left')[['Node', 'Chr', 'Pos', 'Start', 'End', 'Allele', 'CN']]
    logger.info('Total number of segments after filtering: {}'.format(len(filtered_nodes[['Chr', 'Pos', 'Allele']].drop_duplicates())))
    return filtered_nodes, filter_nodes[['Node', 'Chr', 'Pos', 'Start', 'End', 'filter_seg', 'Allele', 'CN']], pos_mapping
    
def get_best_filter_segs(leaves_segments, n_segs, prop, keep_consec):
    leaves_segments_pos = leaves_segments.copy()
    # Select the top n_segs segments and the segments that are the max in the chr to keep, the rest are set to filter.
    leaves_segments_pos['filter_seg'] = np.where(leaves_segments_pos['size_rank'] <= round(n_segs * prop), False, ~leaves_segments_pos['chr_max_seg'])
    leaves_segments_pos = leaves_segments_pos.sort_values(by=['Allele', 'Chr', 'Start', 'End'])
    if keep_consec:
        # Get groups of consecutive segments that have the same filter status
        leaves_segments_pos['group'] = (leaves_segments_pos['filter_seg'] != leaves_segments_pos['filter_seg'].shift(1)).cumsum()
        # Mark the segments from each consecutive group to keep (that were not already being kept) - this is done by selecting the fewest segments possible such that 3 consecutive segments are not filtered, where ties are broken by selecting the largest segments.
        leaves_segments_pos['filter_seg'] = np.where(~leaves_segments_pos['filter_seg'], False, leaves_segments_pos.groupby('group')['seg_size'].transform(lambda x: get_keep_idx(x)))
    return leaves_segments_pos[['Chr', 'Pos', 'Allele', 'Start', 'End', 'seg_size', 'filter_seg']].copy()

def get_keep_idx(x):
    keep = np.zeros(len(x))
    keep[get_min_idx(x)::3] = 1
    return keep.astype(bool)

get_min_idx = (lambda x: min(((i, x[i::3]) for i in range(3)),  key=(lambda p: len(p[1]) + (1 - (sum(p[1]) / sum(x)))))[0])


def phase_profiles(profiles, leaf_dependent=True):
    if type(profiles) == pd.DataFrame:
        input_type = 'df'
        profiles = cn_df_to_dict(profiles, convert_leaf_names=False)
    else:
        input_type = 'dict'
    chromosomes = set(ch for prof in profiles for ch in profiles[prof])
    nodes = set(profiles.keys())
    chrsize = {ch : {len(profiles[prof][ch][0]) for prof in profiles} for ch in chromosomes}
    assert all(len(chrsize[ch]) == 1 for ch in chrsize), 'Chromosomes have different lengths in different leaves'
    chrsize = {ch : chrsize[ch].pop() for ch in chromosomes}
    ref = {node : {ch : tuple([profiles[node][ch][i][0] for i in range(2)]) for ch in chromosomes} for node in nodes}
    
    switch_alleles = (lambda node, ch, : (result[node][ch][0][0], result[node][ch][1][0]) != ref[node][ch])
    swap_profiles = (lambda p, switch: [p[1], p[0]] if switch else p)
    
    if leaf_dependent:
        mkinput = (lambda ch : {b : {node : (profiles[node][ch][0][b], 
                                            profiles[node][ch][1][b]) 
                                    for node in nodes} for b in range(chrsize[ch])})
        phased =  {ch : phasing(mkinput(ch)) for ch in chromosomes}
        result = {node : {ch : [[phased[ch][b][node][0] for b in range(chrsize[ch])],
                                [phased[ch][b][node][1] for b in range(chrsize[ch])]] 
                          for ch in chromosomes} 
                  for node in nodes}
        ld_switchtoref = {ch : np.any([switch_alleles(node, ch) for node in nodes]) for ch in chromosomes}
        result = {node : {ch : swap_profiles(result[node][ch], ld_switchtoref[ch]) 
                          for ch in chromosomes} 
                  for node in nodes}
    else:
        mkinput = (lambda node, ch : {b : {node : (profiles[node][ch][0][b], 
                                                  profiles[node][ch][1][b])} 
                                           for b in range(chrsize[ch])})
        phased = {node : {ch : phasing(mkinput(node, ch)) for ch in chromosomes} for node in nodes}
        result = {node : {ch : [[phased[node][ch][b][node][0] for b in range(chrsize[ch])],
                                [phased[node][ch][b][node][1] for b in range(chrsize[ch])]] 
                          for ch in chromosomes} for node in nodes}
        result = {node : {ch : swap_profiles(result[node][ch], switch_alleles(node, ch)) 
                          for ch in chromosomes} 
                  for node in nodes}
    if input_type == 'dict':
        return cn_dict_to_df(result)
    else:    
        return result
    

def cn_dict_to_df(cn_dict):
    cn_df = pd.DataFrame([{'Node': node, 'Chr' : ch, 'Allele' : ['A', 'B'][allele], 'Pos' : pos+1, 'CN' : cn} for node in cn_dict
                                                                                                for ch in cn_dict[node]
                                                                                                for allele in range(len(cn_dict[node][ch]))
                                                                                                for pos, cn in enumerate(cn_dict[node][ch][allele])])
    return cn_df


def cn_df_to_dict(cn_df, convert_leaf_names=True, allele_specific=True):
    cn_dict = {node : {ch : {allele : cn_df[(cn_df.Node == node) & (cn_df.Chr == ch) & (cn_df.Allele == ['A', 'B'][allele])].CN.values.tolist() for allele in range(1+allele_specific)} for ch in cn_df.Chr.unique()} for node in cn_df.Node.unique()}
    if convert_leaf_names:
        nodes = list(cn_dict.keys())
        cn_dict = {i + 1 : profile for i, (node, profile) in enumerate(cn_dict.items())}
        leaf_mapping = {i + len(nodes) : node for i, node in enumerate(nodes)}
        return cn_dict, leaf_mapping
    else:
        return cn_dict
    
    
    

    