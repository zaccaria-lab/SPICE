"""Pyomo sets and variables for the SPICE multiple-solutions (enumeration) ILP.

Sets
----
V    : all tree nodes (the root is node 0)
V1   : non-root nodes, i.e. nodes with an incoming edge
L    : leaf nodes (observed copy number profiles)
I    : internal non-root nodes
E    : tree edges (parent, child)
N    : genome positions (allele, chr, seg), with seg 1-based
PQ   : genomic intervals (allele, chr, p, q), p <= q, on which copy number events act
O    : pairs of intervals on the same allele and chromosome that overlap or abut
       (allele, chr, p1, q1, p2, q2); used to stop a single event being split in two

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
d, g                            : pre-WGD loss / gain amplitude at each position
dddot, gddot                    : post-WGD loss / gain amplitude at each position
dhat, ghat, dddothat, gddothat  : binary, 1 if that event affects the position
dbar, gbar, dddotbar, gddotbar  : amplitude of an interval event over PQ
*barhat                         : binary, 1 if that interval event occurs
w : 1 if a WGD occurs on the edge into j (Param when fixed, otherwise Var)
z : linearisation of w[j] * cdot[j], i.e. the copies added by the WGD
"""

import logging
from typing import Optional, Sequence

import pyomo.environ as pe
import numpy as np

from spice.libs.ilp.utils import get_seg_pairs, _get_init_values


def add_model_sets(model: pe.ConcreteModel, leaves: dict, edges: Sequence[tuple],
                   logger: logging.Logger) -> tuple[pe.ConcreteModel, int, bool]:
    """Add the tree and genome index sets to the model.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model to add the sets to (modified in place).
    leaves : dict
        Leaf copy number profiles in the format {leaf: {chr: {allele: [cn per segment]}}}.
    edges : list of tuple
        Tree edges as (parent, child) node IDs. The root must be node 0.
    logger : logging.Logger
        Logger used to warn when a chromosome has too many segment pairs.

    Returns
    -------
    model : pyomo.environ.ConcreteModel
        The model with sets V, V1, L, I, E, N, ZERO, N0, PQ and O added.
    n_loci : int
        Number of genome positions (size of N).
    extra_segs : bool
        True if any chromosome allele exceeded 1000 segment pairs and was reduced to single-segment intervals.
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
    single_seg_chrs = []
    seg_pairs_index = [(a, ch, p, q)
                       for (ch, a), segs in chr_index.items()
                       for (p, q) in get_seg_pairs(segs, ch, a, single_seg_chrs, logger)]
    n_loci = len(genome_index)
    zeros_pos = [(allele, chromosome, 0)
                 for chromosome in leaves[leaf_example]
                 for allele in leaves[leaf_example][chromosome]]
    model.ZERO = pe.Set(initialize=zeros_pos)
    model.N = pe.Set(initialize=genome_index)
    model.N0 = pe.Set(initialize=zeros_pos + genome_index)
    model.PQ = pe.Set(initialize=seg_pairs_index)
    overlapping_intervals = [(a1, ch1, p1, q1, p2, q2)
                             for a1, ch1, p1, q1 in model.PQ
                             for a2, ch2, p2, q2 in model.PQ
                             if (a1 == a2) and (ch1 == ch2) and ((ch1, a1) not in single_seg_chrs)
                             and (p1 < p2) and (q1 < q2) and (p2 <= q1 + 1)]
    model.O = pe.Set(initialize=overlapping_intervals)
    model.E = pe.Set(initialize=edges)
    model.L = pe.Set(initialize=list(leaves.keys()))
    model.I = pe.Set(initialize=[i for i in parents if i != root])
    return model, n_loci, len(single_seg_chrs) > 0


def add_copyprofile_variables(model: pe.ConcreteModel, max_cn: int,
                              initialize_model: Optional[pe.ConcreteModel] = None,
                              initialize_model_variables: Optional[list[str]] = None) -> pe.ConcreteModel:
    """Add copy number profile variables c, cbar, chat, cdot, cddot, cddotbar and cddothat.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets V, V1 and N already defined (see ``add_model_sets``).
    max_cn : int
        Upper bound on integer copy number variables.
    initialize_model : pyomo.environ.ConcreteModel, optional
        Solved model (e.g. the optimal ILP) to warm-start variable values from.
    initialize_model_variables : list of str, optional
        Names of variables in ``initialize_model`` to take start values from.

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the copy number variables added.
    """
    if initialize_model_variables is None:
        initialize_model_variables = []
    assert (initialize_model is not None) or (len(initialize_model_variables) == 0), \
        'initialize_model must be provided when initialize_model_variables is non-empty'
    v_n_index = [(j, a, ch, n) for j in model.V for a, ch, n in model.N]
    v1_n_index = [(j, a, ch, n) for j in model.V1 for a, ch, n in model.N]
    # Base
    initialize_c = _get_init_values(initialize_model, initialize_model_variables, 'c', v_n_index)
    initialize_cbar = _get_init_values(initialize_model, initialize_model_variables, 'cbar', v1_n_index)
    initialize_chat = _get_init_values(initialize_model, initialize_model_variables, 'chat', v_n_index)
    model.c = pe.Var(model.V, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn), initialize=initialize_c)
    model.cbar = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn),
                        initialize=initialize_cbar)
    model.chat = pe.Var(model.V, model.N, domain=pe.Binary, initialize=initialize_chat)
    # WGD
    initialize_cdot = _get_init_values(initialize_model, initialize_model_variables, 'cdot', v1_n_index)
    initialize_cddot = _get_init_values(initialize_model, initialize_model_variables, 'cddot', v1_n_index)
    initialize_cddotbar = _get_init_values(initialize_model, initialize_model_variables, 'cddotbar', v1_n_index)
    initialize_cddothat = _get_init_values(initialize_model, initialize_model_variables, 'cddothat', v1_n_index)
    model.cdot = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn),
                        initialize=initialize_cdot)
    model.cddot = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn),
                         initialize=initialize_cddot)
    model.cddotbar = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, bounds=(0, max_cn),
                            initialize=initialize_cddotbar)
    model.cddothat = pe.Var(model.V1, model.N, domain=pe.Binary, initialize=initialize_cddothat)
    return model


def add_event_variables(model: pe.ConcreteModel, max_cn: int) -> pe.ConcreteModel:
    """Add per-position gain and loss variables g, d, ghat, dhat (pre-WGD) and gddot, dddot, gddothat, dddothat
    (post-WGD).

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets V1 and N already defined (see ``add_model_sets``).
    max_cn : int
        Upper bound on event amplitude.

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the event variables added.
    """
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


def add_interval_event_variables(model: pe.ConcreteModel, max_cn: int) -> pe.ConcreteModel:
    """Add interval event variables over PQ: gbar, dbar, gbarhat, dbarhat (pre-WGD) and gddotbar, dddotbar,
    gddotbarhat, dddotbarhat (post-WGD).

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets V1 and PQ already defined (see ``add_model_sets``).
    max_cn : int
        Upper bound on interval event amplitude.

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the interval event variables added.
    """
    # Base
    model.gbar = pe.Var(model.V1, model.PQ, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.dbar = pe.Var(model.V1, model.PQ, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.gbarhat = pe.Var(model.V1, model.PQ, domain=pe.Binary)
    model.dbarhat = pe.Var(model.V1, model.PQ, domain=pe.Binary)
    # WGD
    model.gddotbar = pe.Var(model.V1, model.PQ, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.dddotbar = pe.Var(model.V1, model.PQ, domain=pe.NonNegativeIntegers, bounds=(0, max_cn))
    model.gddotbarhat = pe.Var(model.V1, model.PQ, domain=pe.Binary)
    model.dddotbarhat = pe.Var(model.V1, model.PQ, domain=pe.Binary)
    return model


def add_wgd_variables(model: pe.ConcreteModel, initialize_model: Optional[pe.ConcreteModel] = None,
                      initialize_model_variables: Optional[list[str]] = None,
                      fix_wgd: bool = True) -> tuple[pe.ConcreteModel, Optional[dict]]:
    """Add whole-genome duplication variables w and z.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets V1 and N already defined (see ``add_model_sets``).
    initialize_model : pyomo.environ.ConcreteModel, optional
        Solved model to take start values for w and z from.
    initialize_model_variables : list of str, optional
        Names of variables in ``initialize_model`` to take start values from.
    fix_wgd : bool, optional
        If True, w is a fixed Param taken from ``initialize_model``; otherwise it is a free binary Var
        (default: True).

    Returns
    -------
    model : pyomo.environ.ConcreteModel
        The model with w and z added.
    initialize_w : dict or None
        WGD start values {node: 0 or 1}, or None if w was not initialised.
    """
    if initialize_model_variables is None:
        initialize_model_variables = []
    assert (initialize_model is not None) or (len(initialize_model_variables) == 0), \
        'initialize_model must be provided when initialize_model_variables is non-empty'
    assert (not fix_wgd) or ('w' in initialize_model_variables), \
        'fix_wgd=True requires w values from initialize_model (w is a fixed parameter)'
    v1_n_index = [(j, a, ch, n) for j in model.V1 for a, ch, n in model.N]
    initialize_w = _get_init_values(initialize_model, initialize_model_variables, 'w', list(model.V1))
    if initialize_w is not None:
        initialize_w = {**{j: 0 for j in model.V1}, **initialize_w}
    initialize_z = _get_init_values(initialize_model, initialize_model_variables, 'z', v1_n_index)
    if fix_wgd:
        model.w = pe.Param(model.V1, domain=pe.Binary, initialize=initialize_w)
    else:
        model.w = pe.Var(model.V1, domain=pe.Binary, initialize=initialize_w)
    model.z = pe.Var(model.V1, model.N, domain=pe.NonNegativeIntegers, initialize=initialize_z)
    return model, initialize_w
