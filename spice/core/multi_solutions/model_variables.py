import pyomo.environ as pe
import numpy as np

from spice.core.auxillary.ilp_functions import *

def add_model_sets(model, leaves, allele_specific, edges, logger):
    n_leaves = len(leaves)
    model.V = pe.Set(initialize=range(2*n_leaves))
    model.V1 = pe.Set(initialize=range(1, 2*n_leaves))
    chr_index = {(chromosome, a) : list(np.arange(1, len(profile[a])+1)) for chromosome, profile in leaves[1].items() for a in range(len(profile))}
    genome_index = [(a, ch, n) for (ch, a), segs in chr_index.items() for n in segs]
    single_seg_chrs = []
    seg_codes = []
    seg_pairs_index = [(a, ch, p, q) for (ch, a), segs in chr_index.items() for (p, q) in get_seg_pairs(segs, leaves, ch, a, single_seg_chrs, seg_codes, logger, False)]
    n_loci = len(genome_index)
    zeros_pos = [(allele, chromosome, 0) for chromosome in leaves[1] for allele in leaves[1][chromosome]]
    model.ZERO = pe.Set(initialize=zeros_pos)
    model.N = pe.Set(initialize=genome_index)
    model.N0 = pe.Set(initialize=zeros_pos + genome_index)
    model.PQ = pe.Set(initialize=seg_pairs_index) 
    overlapping_intervals = [((a1, ch1, p1, q1, p2, q2)) for a1, ch1, p1, q1 in model.PQ for a2, ch2, p2, q2 in model.PQ if (a1 == a2) & (ch1 == ch2) & 
                                                                                                                            ((ch1, a1) not in single_seg_chrs) &
                                                                                                                            (p1 < p2) & (q1 < q2) & (p2 <= q1+1)]    
    model.O = pe.Set(initialize=overlapping_intervals)
    model.E = pe.Set(initialize=edges)
    model.L = pe.Set(initialize=range(n_leaves, 2*n_leaves))
    model.I = pe.Set(initialize=range(1, n_leaves))
    return model, n_loci, sum(seg_codes) > 0


# 1.0.1 Copy number profile variables
def add_copyprofile_variables(model, max_cn, initialize_model=None, initialize_model_variables=[]):
    # Base
    initialize_c = None  if 'c' not in initialize_model_variables else ensure_initialize_integers(initialize_model.c.extract_values(), [(j, a, ch, n) for j in model.V for a, ch, n in model.N])
    initialize_cbar = None if 'cbar' not in initialize_model_variables else ensure_initialize_integers(initialize_model.cbar.extract_values(), [(j, a, ch, n) for j in model.V1 for a, ch, n in model.N])
    initialize_chat = None if 'chat' not in initialize_model_variables else ensure_initialize_integers(initialize_model.chat.extract_values(), [(j, a, ch, n) for j in model.V for a, ch, n in model.N])
    model.c = pe.Var(model.V, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn), initialize=initialize_c)
    model.cbar = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn), initialize=initialize_cbar)
    model.chat = pe.Var(model.V, model.N, domain=pe.Binary, initialize=initialize_chat)
    # WGD
    initialize_cdot = None if 'cdot' not in initialize_model_variables else ensure_initialize_integers(initialize_model.cdot.extract_values(), [(j, a, ch, n) for j in model.V1 for a, ch, n in model.N])
    initialize_cddot = None if 'cddot' not in initialize_model_variables else ensure_initialize_integers(initialize_model.cddot.extract_values(), [(j, a, ch, n) for j in model.V1 for a, ch, n in model.N])
    initialize_cddotbar = None if 'cddotbar' not in initialize_model_variables else ensure_initialize_integers(initialize_model.cddotbar.extract_values(), [(j, a, ch, n) for j in model.V1 for a, ch, n in model.N])
    initialize_cddothat = None if 'cddothat' not in initialize_model_variables else ensure_initialize_integers(initialize_model.cddothat.extract_values(), [(j, a, ch, n) for j in model.V1 for a, ch, n in model.N])
    model.cdot = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn), initialize=initialize_cdot)
    model.cddot = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn), initialize=initialize_cddot)
    model.cddotbar = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn), initialize=initialize_cddotbar)
    model.cddothat = pe.Var(model.V1, model.N, domain=pe.Binary, initialize=initialize_cddothat)
    return model


# 1.0.2 Copy number event variables
def add_event_variables(model, max_cn, initialize_model=None, initialize_model_variables=[]):
    # Base
    model.g = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.d = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.ghat = pe.Var(model.V1, model.N, domain=pe.Binary)
    model.dhat = pe.Var(model.V1, model.N, domain=pe.Binary)
    # WGD
    model.gddot = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.dddot = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.gddothat = pe.Var(model.V1, model.N, domain=pe.Binary)
    model.dddothat = pe.Var(model.V1, model.N, domain=pe.Binary)
    return model


def add_interval_event_variables(model, max_cn):
    # Base
    model.gbar = pe.Var(model.V1, model.PQ, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.dbar = pe.Var(model.V1, model.PQ, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.gbarhat = pe.Var(model.V1, model.PQ, domain=pe.Binary)
    model.dbarhat = pe.Var(model.V1,model.PQ, domain=pe.Binary)
    # WGD
    model.gddotbar = pe.Var(model.V1, model.PQ, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.dddotbar = pe.Var(model.V1, model.PQ, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.gddotbarhat = pe.Var(model.V1, model.PQ, domain=pe.Binary)
    model.dddotbarhat = pe.Var(model.V1, model.PQ, domain=pe.Binary)
    return model


# 1.0.3 Whole-genome duplication variables
def add_wgd_variables(model, initialize_model=None, initialize_model_variables=[], fix_wgd=True):
    initialize_w = None if 'w' not in initialize_model_variables else {**{j : 0 for j in model.V1}, **ensure_initialize_integers(initialize_model.w.extract_values(), [(j) for j in model.V1])}
    initialize_z = None if 'z' not in initialize_model_variables else ensure_initialize_integers(initialize_model.z.extract_values(), [(j, a, ch, n) for j in model.V1 for a, ch, n in model.N])
    if fix_wgd:
        model.w = pe.Param(model.V1, domain=pe.Binary, initialize=initialize_w) 
    else:
        model.w = pe.Var(model.V1, domain=pe.Binary, initialize=initialize_w)
    model.z = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, initialize=initialize_z)
    return model, initialize_w



ensure_initialize_integers = (lambda model_var, keys : {k : round(v) if v is not None else v for k, v in model_var.items() if k in keys})

