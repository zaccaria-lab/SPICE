"""Convert copy number profiles between long-format DataFrames and nested dicts.

SPICE reads leaf profiles as a long-format DataFrame (one row per node, chromosome, allele and bin), while the
tree bootstrapping and the ILP use nested dicts of the form {node: {chr: {allele: [cn per segment]}}}, with
alleles keyed 0 ('A') and 1 ('B'). It also maps SPICE segment positions back to the input bins.

cn_dict_to_df : nested dict -> DataFrame
cn_df_to_dict : DataFrame -> nested dict, optionally renaming nodes
map_to_original_positions : interval events at a segment level -> input bin positions and coordinates
expand_segments : per-segment table -> one row per smaller segment (merged segment or input bin)
sort_by_columns : sort output DataFrames by the given columns, comparing numbers inside text as numbers
"""

from typing import Optional

import pandas as pd
import re


def cn_dict_to_df(cn_dict: dict) -> pd.DataFrame:
    """Convert nested copy number profiles into a long-format DataFrame.

    Parameters
    ----------
    cn_dict : dict
        Copy number profiles in the format {node: {chr: {allele: [cn per segment]}}}, with alleles keyed 0 and 1
        (a list [allele 0, allele 1] per chromosome also works).

    Returns
    -------
    cn_df : pd.DataFrame
        One row per node, chromosome, allele and segment, with columns 'Node', 'Chr', 'Allele' ('A' for allele 0,
        'B' for allele 1), 'Pos' (segment number, from 1 within each chromosome and allele) and 'CN'. Rows follow
        the order of `cn_dict`.
    """
    cn_df = pd.DataFrame([{'Node': node, 'Chr': ch, 'Allele': ['A', 'B'][allele], 'Pos': pos + 1, 'CN': cn}
                          for node in cn_dict
                          for ch in cn_dict[node]
                          for allele in range(len(cn_dict[node][ch]))
                          for pos, cn in enumerate(cn_dict[node][ch][allele])])
    return cn_df


def cn_df_to_dict(cn_df: pd.DataFrame, node_conversion: Optional[dict] = None, allele_specific: bool = True) -> dict:
    """Convert long-format copy number profiles into nested dicts.

    Parameters
    ----------
    cn_df : pd.DataFrame
        Copy number profiles, one row per node, chromosome, allele and segment, with columns 'Node', 'Chr', 'Allele'
        ('A' or 'B') and 'CN', with rows in genome order within each chromosome and allele.
    node_conversion : dict, optional
        Mapping from the node names in `cn_df` to the keys of the returned dict. If None, the names are kept.
    allele_specific : bool, default True
        If True, each allele's copy number is modelled separately and both alleles 'A' and 'B' are read. If False,
        total copy number is modelled and only rows with 'Allele' 'A' are read.

    Returns
    -------
    cn_dict : dict
        Copy number profiles in the format {node: {chr: {allele: [cn per segment]}}}, with alleles keyed 0 ('A')
        and 1 ('B').
    """
    if node_conversion is None:
        node_conversion = {node: node for node in cn_df.Node.unique()}
    cn_dict = {node_conversion[node]: {ch: {allele: cn_df[(cn_df.Node == node) &
                                                          (cn_df.Chr == ch) &
                                                          (cn_df.Allele == ['A', 'B'][allele])].CN.values.tolist()
                                            for allele in range(1 + allele_specific)}
                                       for ch in cn_df.Chr.unique()}
               for node in cn_df.Node.unique()}
    empty = [(node, ch, allele) for node in cn_dict for ch in cn_dict[node] for allele in cn_dict[node][ch]
             if len(cn_dict[node][ch][allele]) == 0]
    assert len(empty) == 0, (f'No copy numbers found for {len(empty)} (node, chr, allele), e.g. {empty[:3]}; '
                             "check the 'Allele' column")
    return cn_dict


def map_to_original_positions(events: pd.DataFrame, pos_mapping: pd.DataFrame, level: str) -> pd.DataFrame:
    """Map the 'startPos' / 'endPos' of interval events from segments at `level` back to the input bins.

    Parameters
    ----------
    events : pd.DataFrame
        One row per event, with 'Chr', 'Allele' (0 or 1) and the first and last segment of the event at `level` in
        'startPos' / 'endPos'.
    pos_mapping : pd.DataFrame
        Position mapping from `process_spice_input`, with 'Allele' mapped to 0 / 1: one row per input bin, chromosome
        and allele, with 'origPos', `level`, 'Start' and 'End'.
    level : str
        Segment level of 'startPos' / 'endPos': 'finalPos' (ILP segments) or 'mergePos' (segments after the first
        merge).

    Returns
    -------
    pd.DataFrame
        `events` with 'startPos' / 'endPos' replaced by the event's first and last input bin, and its 'Start' / 'End'
        (the start of the first bin and the end of the last) added. The index is reset.
    """
    bounds = (pos_mapping
              .groupby(['Chr', 'Allele', level])
              .agg(firstPos=('origPos', 'min'), lastPos=('origPos', 'max'), Start=('Start', 'min'),
                   End=('End', 'max')))
    return (events
            .join(bounds[['firstPos', 'Start']], on=['Chr', 'Allele', 'startPos'])
            .join(bounds[['lastPos', 'End']], on=['Chr', 'Allele', 'endPos'])
            .drop(columns=['startPos', 'endPos'])
            .rename(columns={'firstPos': 'startPos', 'lastPos': 'endPos'})
            .reset_index(drop=True))


def expand_segments(nodes: pd.DataFrame, pos_mapping: pd.DataFrame, level: str,
                    to_level: str = 'origPos') -> pd.DataFrame:
    """Repeat each row of a per-segment table for every smaller segment it contains.

    Parameters
    ----------
    nodes : pd.DataFrame
        One row per segment at `level` (in 'Pos'), with 'Chr', and 'Allele' (0 or 1) if `level` is 'finalPos'. Any
        'Start' / 'End' columns are replaced.
    pos_mapping : pd.DataFrame
        Position mapping from `process_spice_input`, with 'Allele' mapped to 0 / 1: one row per input bin, chromosome
        and allele, with 'origPos', 'mergePos', 'finalPos', 'Start' and 'End'.
    level : str
        Segment level of 'Pos': 'finalPos' (ILP segments, separate per allele) or 'mergePos' (segments after the
        first merge, shared by both alleles).
    to_level : str, default 'origPos'
        Segment level to expand to: 'mergePos' or 'origPos' (the input bins).

    Returns
    -------
    pd.DataFrame
        One row per row of `nodes` and segment at `to_level` within it, with that segment in 'Pos' and its 'Start' /
        'End' after it.
    """
    keys = ['Chr', 'Allele'] if level == 'finalPos' else ['Chr']
    segs = (pos_mapping
            .groupby(keys + [level, to_level], as_index=False)
            .agg(Start=('Start', 'min'), End=('End', 'max'))
            .rename(columns={level: 'Pos'}))
    cols = [i for i in nodes.columns if i not in ['Start', 'End']]
    cols = cols[:cols.index('Pos') + 1] + ['Start', 'End'] + cols[cols.index('Pos') + 1:]
    return (nodes
            .drop(columns=['Start', 'End'], errors='ignore')
            .merge(segs, on=keys + ['Pos'], how='left')
            .drop(columns='Pos')
            .rename(columns={to_level: 'Pos'})
            [cols])


def sort_by_columns(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sort by the columns in `by` that exist, comparing numbers inside text as numbers ('m2' before 'm10')."""
    split_numbers = (lambda x: tuple(int(p) if p.isdigit() else p for p in re.split(r'(\d+)', str(x))))
    by = [i for i in by if i in df.columns]
    return (df.sort_values(by=by, key=lambda col: col.map(split_numbers) if col.dtype == object else col,
                           kind='stable')
            .reset_index(drop=True))
