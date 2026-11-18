"""Pyomo sets and variables for the SPICE optimal ILP.

Sets
----
V    : all tree nodes (the root is node 0)
V1   : non-root nodes, i.e. nodes with an incoming edge
L    : leaf nodes (observed copy number profiles)
I    : internal non-root nodes
E    : tree edges (parent, child)
N    : genome positions (allele, chr, seg), with seg 1-based
ZERO : one dummy position (allele, chr, 0) per chromosome allele, before its first segment
N0   : ZERO + N; event amplitudes are fixed to 0 at the dummy position, so an event that covers segment 1
       counts as starting there

Variables
---------
On the edge into node j, events happen in this order:
pre-WGD losses -> pre-WGD gains -> WGD -> post-WGD losses -> post-WGD gains.

c         : copy number of node j
cbar      : parent copy number after pre-WGD losses
cdot      : after pre-WGD gains
cddot     : after WGD
cddotbar  : after post-WGD losses (c = cddotbar + gddot)
chat, cddothat                  : binary, 1 if c / cddot is non-zero
d, g                            : pre-WGD loss / gain amplitude at each position (over N0)
dddot, gddot                    : post-WGD loss / gain amplitude at each position (over N0); only allowed on an
                                  edge with a WGD
dhat, ghat, dddothat, gddothat  : binary, 1 if that event affects the position
dbar, gbar, dddotbar, gddotbar  : number of events of that type starting at the position, i.e. at least the rise
                                  in amplitude from the previous position; these are the counts in the objective
w : binary, 1 if a WGD occurs on the edge into j
z : linearisation of w[j] * cdot[j], i.e. the copies added by the WGD
"""

from typing import Optional, Sequence

import pyomo.environ as pe
import numpy as np


def add_model_sets(model: pe.ConcreteModel, leaves: dict, edges: Sequence[tuple]) -> tuple[pe.ConcreteModel, int]:
    """Add the tree and genome index sets to the model.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model to add the sets to (modified in place).
    leaves : dict
        Leaf copy number profiles in the format {leaf: {chr: {allele: [cn per segment]}}}.
    edges : list of tuple
        Tree edges as (parent, child) node IDs. The root must be node 0.

    Returns
    -------
    model : pyomo.environ.ConcreteModel
        The model with sets V, V1, ZERO, N, N0, E, L and I added.
    n_loci : int
        Number of genome positions (size of N).
    """
    assert len(edges) > 0, 'edges must contain at least one (parent, child) edge'
    assert all(len(e) == 2 for e in edges), 'Each edge must be a (parent, child) tuple'
    # Define tree variables
    nodes = list(set([i for j in edges for i in j]))
    parents, children = list(zip(*edges))
    assert len(children) == len(set(children)), 'Each node must have exactly one parent (edges do not form a tree)'
    parents = list(set(parents))
    roots = [i for i in parents if i not in children]
    assert len(roots) == 1, f'Tree must have exactly one root, found {len(roots)}: {roots}'
    root = roots[0]
    assert root == 0, f'Root node must be 0 (constraints assume node 0 is the diploid root), found {root}'
    tips = set(nodes) - set(parents)
    assert set(leaves) == tips, f'Leaf profiles do not match tree leaves: {set(leaves) ^ tips}'
    leaf_example = list(leaves.keys())[0]
    ref_shape = {ch: {a: len(cns) for a, cns in alleles.items()} for ch, alleles in leaves[leaf_example].items()}
    assert all({ch: {a: len(cns) for a, cns in alleles.items()} for ch, alleles in profile.items()} == ref_shape
               for profile in leaves.values()), \
        'All leaves must have the same chromosomes, alleles and number of segments'
    # Create tree variables
    model.V = pe.Set(initialize=nodes)
    model.V1 = pe.Set(initialize=[i for i in nodes if i != root])
    chr_index = {(chromosome, a): list(np.arange(1, len(profile[a]) + 1))
                 for chromosome, profile in leaves[leaf_example].items()
                 for a in range(len(profile))}
    genome_index = [(a, ch, n) for (ch, a), segs in chr_index.items() for n in segs]
    n_loci = len(genome_index)
    zeros_pos = [(allele, chromosome, 0)
                 for chromosome in leaves[leaf_example]
                 for allele in leaves[leaf_example][chromosome]]
    model.ZERO = pe.Set(initialize=zeros_pos)
    model.N = pe.Set(initialize=genome_index)
    model.N0 = pe.Set(initialize=zeros_pos + genome_index)
    model.E = pe.Set(initialize=edges)
    model.L = pe.Set(initialize=list(leaves.keys()))
    model.I = pe.Set(initialize=[i for i in parents if i != root])
    return model, n_loci


def add_copyprofile_variables(model: pe.ConcreteModel, max_cn: int,
                              init_model: Optional[dict] = None) -> pe.ConcreteModel:
    """Add copy number profile variables c, cbar, chat, cdot, cddot, cddotbar and cddothat.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets V, V1, N and I already defined (see ``add_model_sets``).
    max_cn : int
        Upper bound on integer copy number variables.
    init_model : dict, optional
        Start values for c in the format {(node, allele, chr, seg): cn}, e.g. internal node profiles from MEDICC
        (see ``process_medicc_initialise``). If None, c starts at 1 for every internal non-root node (set I).

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the copy number variables added.
    """
    # Base
    if init_model is None:
        initialize_c = {(j, a, ch, n): 1 for j in model.V for a, ch, n in model.N if j in model.I}
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


def add_event_variables(model: pe.ConcreteModel, max_cn: int) -> pe.ConcreteModel:
    """Add per-position gain and loss variables g, d, ghat, dhat (pre-WGD) and gddot, dddot, gddothat, dddothat
    (post-WGD), indexed over N0 so that each has a dummy position 0 before the first segment.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets V1 and N0 already defined (see ``add_model_sets``).
    max_cn : int
        Upper bound on event amplitude.

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the event variables added.
    """
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


def add_start_event_variables(model: pe.ConcreteModel, max_cn: int) -> pe.ConcreteModel:
    """Add event start variables gbar, dbar (pre-WGD) and gddotbar, dddotbar (post-WGD): the number of events of
    each type starting at each position. These are the event counts in the objective.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets V1 and N already defined (see ``add_model_sets``).
    max_cn : int
        Upper bound on the number of events starting at a position.

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the event start variables added.
    """
    # Base
    model.gbar = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.dbar = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    # WGD
    model.gddotbar = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.dddotbar = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    return model


def add_wgd_variables(model: pe.ConcreteModel) -> pe.ConcreteModel:
    """Add whole-genome duplication variables w and z.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets V1 and N already defined (see ``add_model_sets``).

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with w (a free binary Var, starting at 0 for every node) and z added.
    """
    initialize_w = {j: 0 for j in model.V1}
    model.w = pe.Var(model.V1, domain=pe.Binary, initialize=initialize_w)
    model.z = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers)
    return model
