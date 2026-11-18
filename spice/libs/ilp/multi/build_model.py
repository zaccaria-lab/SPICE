"""Assemble the SPICE multiple-solutions (enumeration) ILP.

Each function wraps one stage of building the model and calls the builders:

add_model_vars : sets and variables (see ``variables``, which also documents every set and variable)
add_model_cons : constraints (see ``constraints``)
add_model_obj  : objective (see ``objective``)

Used by ``spice.libs.ilp.multi.solve_model.get_multiple_trees``, which calls the three functions in this order.
"""

import logging
from typing import Optional, Sequence

import pyomo.environ as pe

from spice.libs.ilp.multi import variables, constraints, objective


def add_model_vars(model: pe.ConcreteModel, leaves: dict, edges: Sequence[tuple], max_cn: int,
                   init_model: Optional[pe.ConcreteModel] = None, init_model_vars: Optional[list[str]] = None,
                   logger: Optional[logging.Logger] = None) -> tuple[pe.ConcreteModel, int, Optional[dict], bool]:
    """Add all index sets and variables to the model.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Empty model to add the sets and variables to (modified in place).
    leaves : dict
        Leaf copy number profiles in the format {leaf: {chr: {allele: [cn per segment]}}}.
    edges : list of tuple
        Tree edges as (parent, child) node IDs. The root must be node 0.
    max_cn : int
        Upper bound on copy number and event amplitude variables.
    init_model : pyomo.environ.ConcreteModel, optional
        Solved model (e.g. the optimal ILP) to warm-start variable values from.
    init_model_vars : list of str, optional
        Names of variables in ``init_model`` to take start values from.
    logger : logging.Logger, optional
        Logger used to warn when a chromosome has too many segment pairs.

    Returns
    -------
    model : pyomo.environ.ConcreteModel
        The model with sets V, V1, L, I, E, N, ZERO, N0, PQ and O, and variables c, cbar, chat, cdot, cddot,
        cddotbar, cddothat, g, d, ghat, dhat, gddot, dddot, gddothat, dddothat, gbar, dbar, gbarhat, dbarhat,
        gddotbar, dddotbar, gddotbarhat, dddotbarhat, w and z added.
    n_loci : int
        Number of genome positions (size of N).
    init_w : dict or None
        WGD values {node: 0 or 1} taken from ``init_model``, or None if w was not initialised.
    extra_segs : bool
        True if any chromosome allele exceeded 1000 segment pairs and was reduced to single-segment intervals.
    """
    if init_model_vars is None:
        init_model_vars = []
    model, n_loci, extra_segs = variables.add_model_sets(model, leaves, edges, logger)
    # Add variables
    model = variables.add_copyprofile_variables(model, max_cn, init_model, init_model_vars)
    model = variables.add_event_variables(model, max_cn)
    model = variables.add_interval_event_variables(model, max_cn)
    model, init_w = variables.add_wgd_variables(model, init_model, init_model_vars, fix_wgd=True)
    return model, n_loci, init_w, extra_segs


def add_model_cons(model: pe.ConcreteModel, leaves: dict, max_cn: int, allele_specific: bool) -> pe.ConcreteModel:
    """Add all ILP constraints to the model.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with all sets and variables from ``add_model_vars``, and a ``ConstraintList`` at ``model.cons``.
    leaves : dict
        Leaf copy number profiles in the format {leaf: {chr: {allele: [cn per segment]}}}.
    max_cn : int
        Maximum copy number, used as the big-M bound in the linking constraints.
    allele_specific : bool
        If True, model the copy number of each allele separately; if False, model total copy number throughout the
        tree (the root is then fixed to 2 rather than 1 per allele).

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the input profile, profile zero, inheritance, gain and loss event, WGD, interval event and
        cherry constraints added to ``model.cons``.
    """
    model = constraints.add_input_profile_constraints(model, leaves, allele_specific)
    model = constraints.add_profile_zero_constraints(model, max_cn)
    model = constraints.add_inheritance_constraints(model)
    model = constraints.add_gain_and_loss_event_constraints(model, max_cn)
    model = constraints.add_wgd_constraints(model, max_cn)
    model = constraints.add_interval_event_constraints(model, max_cn)
    model = constraints.add_cherry_constraints(model)
    return model


def add_model_obj(model: pe.ConcreteModel, max_cn: int, weight_wgd: bool) -> pe.ConcreteModel:
    """Add the objective: minimise the total event amplitude plus the WGD cost.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with all sets and variables from ``add_model_vars``.
    max_cn : int
        Maximum copy number, used as the cost of a WGD when ``weight_wgd`` is True.
    weight_wgd : bool
        If True, each WGD costs ``max_cn``; if False, it costs 1.

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the objective Cost added.
    """
    model = objective.add_objective_function(model, max_cn, weight_wgd)
    return model
