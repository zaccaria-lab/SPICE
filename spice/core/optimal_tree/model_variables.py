import pyomo.environ as pe
import numpy as np

def add_model_sets(model, leaves, allele_specific, edges):
    # define tree variables
    n_leaves = len(leaves)
    nodes = list(set([i for j in edges for i in j]))
    parents, children = list(zip(*edges))
    parents = list(set(parents))
    root = [i for i in parents if i not in children][0]
    leaf_example = list(leaves.keys())[0]
    # create tree variables
    model.V = pe.Set(initialize=nodes)
    model.V1 = pe.Set(initialize=[i for i in nodes if i != root])
    chr_index = {(chromosome, a) : list(np.arange(1, len(profile[a])+1)) for chromosome, profile in leaves[leaf_example].items() for a in range(len(profile))}
    genome_index = [(a, ch, n) for (ch, a), segs in chr_index.items() for n in segs]
    n_loci = len(genome_index)
    zeros_pos = [(allele, chromosome, 0) for chromosome in leaves[leaf_example] for allele in leaves[leaf_example][chromosome]]
    model.ZERO = pe.Set(initialize=zeros_pos)
    model.N = pe.Set(initialize=genome_index)
    model.N0 = pe.Set(initialize=zeros_pos + genome_index)
    model.E = pe.Set(initialize=edges)
    model.L = pe.Set(initialize=list(leaves.keys()))
    model.I = pe.Set(initialize=[i for i in parents if i != root])
    return model, n_loci


# 1.0.1 Copy number profile variables
def add_copyprofile_variables(model, max_cn, init_model=None):
    # Base
    if init_model is None:
        initialize_c = {(j, a, ch, n) : 1 for j in model.V for a, ch, n in model.N if j in model.I}
    else:
        initialize_c = init_model
    model.c = pe.Var(model.V, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn), initialize=initialize_c)
    model.cbar = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.chat = pe.Var(model.V, model.N, domain=pe.Binary)
    # WGD
    model.cdot = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.cddot = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.cddotbar = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.cddothat = pe.Var(model.V1, model.N, domain=pe.Binary)
    return model


# 1.0.2 Copy number event variables
def add_event_variables(model, max_cn):
    # Base
    model.g = pe.Var(model.V1, model.N0, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.d = pe.Var(model.V1, model.N0, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.ghat = pe.Var(model.V1, model.N0, domain=pe.Binary)
    model.dhat = pe.Var(model.V1, model.N0, domain=pe.Binary)
    # WGD
    model.gddot = pe.Var(model.V1, model.N0, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.dddot = pe.Var(model.V1, model.N0, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.gddothat = pe.Var(model.V1, model.N0, domain=pe.Binary)
    model.dddothat = pe.Var(model.V1, model.N0, domain=pe.Binary)
    return model


def add_start_event_variables(model, max_cn):
    # Variables
    model.gbar = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.dbar = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.gddotbar = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.dddotbar = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    return model


# 1.0.4 Whole-genome duplication variables
def add_wgd_variables(model):
    initialize_w = {(j) : 0 for j in model.V1}
    model.w = pe.Var(model.V1, domain=pe.Binary, initialize=initialize_w) 
    model.z = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers)
    return model






