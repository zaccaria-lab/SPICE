"""Assemble the SPICE optimal ILP.

Each function wraps one stage of building the model and calls the builders:

add_model_vars : sets and variables (see ``variables``, which also documents every set and variable)
add_model_cons : constraints (see ``constraints``)
add_model_obj  : objective (see ``objective``)

Used by ``spice.libs.ilp.optimal.solve_model.get_optimal_tree``, which calls the three functions in this order.
"""

from typing import Optional, Sequence

import pyomo.environ as pe

from spice.libs.ilp.optimal import variables, constraints, objective


def add_model_vars(model: pe.ConcreteModel, leaves: dict, edges: Sequence[tuple], max_cn: int,
                   init_model: Optional[dict]) -> tuple[pe.ConcreteModel, int]:
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
    init_model : dict or None
        Start values for c in the format {(node, allele, chr, seg): cn}, e.g. internal node profiles from MEDICC
        (see ``process_medicc_initialise``). If None, c starts at 1 for every internal non-root node (set I).

    Returns
    -------
    model : pyomo.environ.ConcreteModel
        The model with sets V, V1, ZERO, N, N0, E, L and I, and variables c, cbar, chat, cdot, cddot, cddotbar,
        cddothat, g, d, ghat, dhat, gddot, dddot, gddothat, dddothat, gbar, dbar, gddotbar, dddotbar, w and z added.
    n_loci : int
        Number of genome positions (size of N).
    """
    model, n_loci = variables.add_model_sets(model, leaves, edges)
    # Add variables
    model = variables.add_copyprofile_variables(model, max_cn, init_model=init_model)
    model = variables.add_event_variables(model, max_cn)
    model = variables.add_start_event_variables(model, max_cn)
    model = variables.add_wgd_variables(model)
    return model, n_loci


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
        The model with the input profile, profile zero, inheritance, gain and loss event, WGD, start event position
        and start event constraints added to ``model.cons``.
    """
    model = constraints.add_input_profile_constraints(model, leaves, allele_specific)
    model = constraints.add_profile_zero_constraints(model, max_cn)
    model = constraints.add_inheritance_constraints(model)
    model = constraints.add_gain_and_loss_event_constraints(model, max_cn)
    model = constraints.add_wgd_constraints(model, max_cn)
    model = constraints.add_start_event_pos_constraints(model)
    model = constraints.add_start_event_constraints(model)
    return model


def add_model_obj(model: pe.ConcreteModel, max_cn: int, n_loci: int, weight_wgd: bool,
                  node_weight_dict: dict) -> pe.ConcreteModel:
    """Add the objective: minimise the number of events plus the WGD cost, with a node-weighted amplitude term.

    The main term counts event starts (gbar, dbar, gddotbar, dddotbar) plus the cost of each WGD over all non-root
    nodes. The second term is a subordinate objective that breaks ties in favour of trees with events higher up the
    tree.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with all sets and variables from ``add_model_vars``.
    max_cn : int
        Maximum copy number, used as the cost of a WGD when ``weight_wgd`` is True, and to scale the amplitude term.
    n_loci : int
        Number of genome positions (size of N), from ``add_model_vars``; used to scale the amplitude term.
    weight_wgd : bool
        If True, each WGD costs ``max_cn``; if False, it costs 1.
    node_weight_dict : dict
        Weight of each non-root node in the amplitude term, {node: weight}, e.g. from ``get_node_depth_weighting``.

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the objective Cost added.
    """
    model = objective.add_objective_function(model, max_cn, n_loci, weight_wgd, node_weight_dict=node_weight_dict)
    return model
