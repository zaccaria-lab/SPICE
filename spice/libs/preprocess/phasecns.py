"""Allele phasing of copy number profiles.

In allele-specific input, SPICE provides the option to phase haplotypes if the input labels A and B do not refer to the
same parental allele from one bin to the next (i.e. if a chromosome reads A = 3, B = 1 in one bin and A = 1, B = 3 in
the next). `phase_profiles` reassigns the two copy numbers of each bin between A and B so that they change as little as
possible along each chromosome.

- `phasing` phases one chromosome by dynamic programming over its bins. Each bin is either kept or swapped (its two
  copy numbers exchanged), and the cost between consecutive bins is the number of allele copy numbers that differ,
  summed over the nodes in the table. Backtracking gives the choice per bin with the lowest total cost.
- `phase_profiles` calls `phasing` for each chromosome, either on all leaves together (`leaf_dependent=True`, every
  leaf takes the same choice in a bin) or on each leaf separately. It then swaps whole chromosomes back where needed,
  so that the first bin of each chromosome keeps its input allele order.

Used by `process_spice_input` (`spice/libs/preprocess/input.py`) when `phase_alleles` is True in allele-specific mode.
"""

import pandas as pd
import numpy as np

from spice.libs.utils.conversion import cn_df_to_dict, cn_dict_to_df


def phase_profiles(profiles: pd.DataFrame, leaf_dependent: bool = True) -> pd.DataFrame:
    """Phase the alleles of each chromosome so that copy numbers change as little as possible between bins.

    Parameters
    ----------
    profiles : pandas.DataFrame
        Leaf copy number profiles as a DataFrame with columns 'Node', 'Chr', 'Allele' ('A' or 'B') and 'CN',
        one row per leaf, chromosome, allele and bin, with bins in genome order. Every leaf must have the same number of
        bins on each chromosome, on both alleles.
    leaf_dependent : bool, optional
        If True (default), phase all leaves together, so that every leaf takes the same choice (kept or swapped) in a
        bin. If False, phase each leaf separately.

    Returns
    -------
    pandas.DataFrame
        Phased profiles. A DataFrame with columns 'Node', 'Chr', 'Allele', 'Pos' (1-based bin), 'CN', 'Start' and
        'End', one row per leaf, chromosome, allele and bin.
    """
    genome_locs = profiles[['Chr', 'Allele', 'Pos', 'Start', 'End']].drop_duplicates()
    profiles = cn_df_to_dict(profiles)
    chromosomes = set(ch for prof in profiles for ch in profiles[prof])
    nodes = set(profiles.keys())
    chrsize = {ch: {len(profiles[prof][ch][0]) for prof in profiles} for ch in chromosomes}
    assert all(len(chrsize[ch]) == 1 for ch in chrsize), 'Chromosomes have different lengths in different leaves'
    chrsize = {ch: chrsize[ch].pop() for ch in chromosomes}
    assert all(len(profiles[prof][ch][1]) == chrsize[ch] for prof in profiles for ch in chromosomes), \
        'Alleles A and B have different numbers of bins in some leaves'
    ref = {node: {ch: tuple([profiles[node][ch][i][0] for i in range(2)]) for ch in chromosomes} for node in nodes}

    switch_alleles = lambda node, ch: (result[node][ch][0][0], result[node][ch][1][0]) != ref[node][ch]
    swap_profiles = lambda p, switch: [p[1], p[0]] if switch else p

    if leaf_dependent:
        mkinput = lambda ch: {b: {node: (profiles[node][ch][0][b],
                                         profiles[node][ch][1][b])
                                  for node in nodes} for b in range(chrsize[ch])}
        phased = {ch: phasing(mkinput(ch)) for ch in chromosomes}
        result = {node: {ch: [[phased[ch][b][node][0] for b in range(chrsize[ch])],
                              [phased[ch][b][node][1] for b in range(chrsize[ch])]]
                         for ch in chromosomes}
                  for node in nodes}
        ld_switchtoref = {ch: np.any([switch_alleles(node, ch) for node in nodes]) for ch in chromosomes}
        result = {node: {ch: swap_profiles(result[node][ch], ld_switchtoref[ch])
                         for ch in chromosomes}
                  for node in nodes}
    else:
        mkinput = lambda node, ch: {b: {node: (profiles[node][ch][0][b],
                                               profiles[node][ch][1][b])}
                                    for b in range(chrsize[ch])}
        phased = {node: {ch: phasing(mkinput(node, ch)) for ch in chromosomes} for node in nodes}
        result = {node: {ch: [[phased[node][ch][b][node][0] for b in range(chrsize[ch])],
                              [phased[node][ch][b][node][1] for b in range(chrsize[ch])]]
                         for ch in chromosomes} for node in nodes}
        result = {node: {ch: swap_profiles(result[node][ch], switch_alleles(node, ch))
                         for ch in chromosomes}
                  for node in nodes}
    return cn_dict_to_df(result).merge(genome_locs, on=['Chr', 'Allele', 'Pos'], how='left')


def phasing(table: dict) -> dict:
    """Phase the alleles of one chromosome by dynamic programming over its bins.

    Each bin is either kept or swapped (its two copy numbers exchanged). The cost between consecutive bins is the
    number of allele copy numbers that differ between them, summed over the nodes in the table, and backtracking gives
    the choice per bin with the lowest total cost. The result can be swapped in every bin relative to the input;
    `phase_profiles` restores the input allele order of the first bin.

    Parameters
    ----------
    table : dict
        Copy numbers of one chromosome, {bin: {node: (cn of A, cn of B)}}, for all leaves or for a single leaf. Bins
        are phased in the sorted order of their keys, and every bin must contain every node.

    Returns
    -------
    dict
        Phased copy numbers in the same format, {bin: {node: (cn of A, cn of B)}}.
    """
    D = []
    B = []
    cells = set(e for b in table for e in table[b])
    swapped = lambda p: (p[1], p[0])
    norm = {i: {e: table[b][e] for e in cells} for i, b in enumerate(sorted(table.keys()))}
    swap = {i: {e: swapped(table[b][e]) for e in cells} for i, b in enumerate(sorted(table.keys()))}
    hd = lambda x, y: 1 if x != y else 0
    dist = lambda x, y: sum(hd(x[e][0], y[e][0]) + hd(x[e][1], y[e][1]) for e in cells)

    for i in range(len(table.keys())):
        if i == 0:
            Dnon, Dswa = 0, 0
            Bnon, Bswa = -1, -1
        else:
            Dnonnon = D[i - 1]['0'] + dist(norm[i - 1], norm[i])
            Dswanon = D[i - 1]['1'] + dist(swap[i - 1], norm[i])
            Dnon = min(Dnonnon, Dswanon)
            Bnon = '0' if Dnonnon <= Dswanon else '1'

            Dnonswa = D[i - 1]['0'] + dist(norm[i - 1], swap[i])
            Dswaswa = D[i - 1]['1'] + dist(swap[i - 1], swap[i])
            Dswa = min(Dnonswa, Dswaswa)
            Bswa = '0' if Dnonswa <= Dswaswa else '1'

        D.append({'0': Dnon, '1': Dswa})
        B.append({'0': Bnon, '1': Bswa})

    res = []
    for i in reversed(range(len(table))):
        if i == len(table) - 1:
            if D[i]['0'] < D[i]['1']:
                res.insert(0, norm[i])
                prev = B[i]['0']
            else:
                res.insert(0, swap[i])
                prev = B[i]['1']
        else:
            res.insert(0, norm[i] if prev == '0' else swap[i])
            prev = B[i][prev]

    assert prev == -1, 'The backtracking of the dynamic programming did not terminate with the first bin'
    return {b: res[i] for i, b in enumerate(sorted(table.keys()))}
