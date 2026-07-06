import pandas as pd
import numpy as np

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
    

def phasing(table):
    D = []
    B = []
    cells = set(e for b in table for e in table[b])
    swapped = (lambda p : (p[1], p[0]))
    norm = {i : {e : table[b][e] for e in cells} for i, b in enumerate(sorted(table.keys()))}
    swap = {i : {e : swapped(table[b][e]) for e in cells} for i, b in enumerate(sorted(table.keys()))}
    hd = (lambda x, y : 1 if x != y else 0)
    dist = (lambda x, y : sum(hd(x[e][0], y[e][0]) + hd(x[e][1], y[e][1]) for e in cells))

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

        D.append({'0' : Dnon, '1' : Dswa})
        B.append({'0' : Bnon, '1' : Bswa})

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
    return {b : res[i] for i, b in enumerate(sorted(table.keys()))}


# BSD 3-Clause License

# Copyright (c) 2019, Princeton University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# def phase_leaf_independent(leaves):
#     phased_leaves = {leaf: {ch: phasing_lid(cn_profile) for ch, cn_profile in genome.items()} for leaf, genome in leaves.items()}
#     compare_leaf = phased_leaves[list(leaves.keys())[0]]
#     return {leaf : {ch : order_leaf_alleles(compare_leaf[ch], cns) for ch, cns in genome.items()} for leaf, genome in phased_leaves.items()}


# def phasing_lid(cn_profile):
#     cn_tuples = list(zip(cn_profile[0], cn_profile[1]))
#     tuple_pairs = list(zip(cn_tuples[:-1], cn_tuples[1:]))
#     switch = [phase_pairs(pair) for pair in tuple_pairs]
#     leaf = np.array(cn_profile).T
#     for i, s in enumerate(switch):
#         if s == 1:
#             leaf[i:] = np.flip(leaf[i:], axis=1)
#     return leaf.T.tolist()

# get_breakpoints = (lambda pair: sum([pair[0][i] != pair[1][i] for i in range(2)]))
# get_difference = (lambda pair: sum([abs(pair[0][i] - pair[1][i]) for i in range(2)]))
# swap = (lambda pair: (pair[0], (pair[1][1], pair[1][0])))
# phase_pairs = (lambda pair: min((0, 1), key=(lambda x: (get_breakpoints(pair if x == 0 else swap(pair)), get_difference(pair if x == 0 else swap(pair))))))

# get_leaves_diff = (lambda leaf1cn, leaf2cn : sum(abs(np.array(leaf1cn) - np.array(leaf2cn))))
# order_leaf_alleles = (lambda leaf1, leaf2: [leaf2[0], leaf2[1]] if (get_leaves_diff(leaf1[0], leaf2[0]) + get_leaves_diff(leaf1[1], leaf2[1])) < (get_leaves_diff(leaf1[0], leaf2[1]) + get_leaves_diff(leaf1[1], leaf2[0])) else [leaf2[1], leaf2[0]])

# def phase_leaf_independent(profiles):
#     chromosomes = set(ch for prof in profiles for ch in profiles[prof])
#     nodes = set(profiles.keys())
#     chrsize = {ch : {len(profiles[prof][ch][0]) for prof in profiles} for ch in chromosomes}
#     assert all(len(chrsize[ch]) == 1 for ch in chrsize), 'Chromosomes have different lengths in different leaves'
#     chrsize = {ch : chrsize[ch].pop() for ch in chromosomes}
#     mkinput = (lambda node, ch : {b : {node : (profiles[node][ch][0][b], 
#                                         profiles[node][ch][1][b])} for b in range(chrsize[ch])})
#     phased = {node : {ch : phasing(mkinput(node, ch)) for ch in chromosomes} for node in nodes}
#     return {node : {ch : [[phased[node][ch][b][node][0] for b in range(chrsize[ch])],
#                         [phased[node][ch][b][node][1] for b in range(chrsize[ch])]] for ch in chromosomes} for node in nodes}
    

# def phase_leaf_dependent(profiles):
#     chromosomes = set(ch for prof in profiles for ch in profiles[prof])
#     nodes = set(profiles.keys())
#     chrsize = {ch : {len(profiles[prof][ch][0]) for prof in profiles} for ch in chromosomes}
#     assert all(len(chrsize[ch]) == 1 for ch in chrsize), 'Chromosomes have different lengths in different leaves'
#     chrsize = {ch : chrsize[ch].pop() for ch in chromosomes}
#     mkinput = (lambda ch : {b : {node : (profiles[node][ch][0][b], 
#                                          profiles[node][ch][1][b]) for node in nodes} for b in range(chrsize[ch])})
#     phased =  {ch : phasing(mkinput(ch)) for ch in chromosomes}
#     return {node : {ch : [[phased[ch][b][node][0] for b in range(chrsize[ch])],
#                           [phased[ch][b][node][1] for b in range(chrsize[ch])]] for ch in chromosomes} for node in nodes}

