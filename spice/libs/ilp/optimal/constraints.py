"""Pyomo constraints for the SPICE optimal ILP.

Each function adds one group of constraints to a model that already has the sets and variables from
``spice.libs.ilp.optimal.variables`` and a ``ConstraintList`` at ``model.cons``.

Sets
----
V1   : non-root nodes, i.e. nodes with an incoming edge (the root is node 0)
L    : leaf nodes (observed copy number profiles)
I    : internal non-root nodes
E    : tree edges (parent, child)
N    : genome positions (allele, chr, seg), with seg 1-based
ZERO : one dummy position (allele, chr, 0) per chromosome allele, before its first segment

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
dbar, gbar, dddotbar, gddotbar  : number of events of that type starting at the position
w : binary, 1 if a WGD occurs on the edge into j
z : linearisation of w[j] * cdot[j], i.e. the copies added by the WGD

Constraint groups
-----------------
input profile     : root is diploid (1 per allele, or 2 if not allele-specific); leaves equal the observed profiles
profile zero      : chat / cddothat are 1 exactly when c / cddot is non-zero
inheritance       : prevent re-gaining of copies after LOH
WGD               : z = w * cdot (linearised), cddot = cdot + z; post-WGD events only occur on WGD edges; at most
                    all but one child of an internal node can have a WGD
start event pos   : event amplitudes are 0 at the dummy position before each chromosome allele
gain and loss     : chains parent -> cbar -> cdot -> cddot -> cddotbar -> c through the event amplitudes; the lower
                    bounds are relaxed where the later stage is zero, so events don't affect segments that already
                    have 0 copies and instead span these segments; each position has at most one of gain / loss
                    before the WGD and at most one after
start events      : an event starts at a position wherever its amplitude rises from the previous position; these
                    starts are the event counts in the objective
"""

import pyomo.environ as pe


def add_input_profile_constraints(model: pe.ConcreteModel, leaves: dict, allele_specific: bool) -> pe.ConcreteModel:
    """Fix the root to a diploid profile and the leaves to their observed copy number profiles.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets N and L, variables c and chat, and ``model.cons`` already defined.
    leaves : dict
        Leaf copy number profiles in the format {leaf: {chr: {allele: [cn per segment]}}}.
    allele_specific : bool
        If True, the root has copy number 1 on each allele; otherwise its total copy number is 2.

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the input profile constraints added to ``model.cons``.
    """
    for a, ch, s in model.N:
        if not allele_specific:
            model.cons.add(expr=model.c[0, a, ch, s] == 2)
        else:
            model.cons.add(expr=model.c[0, a, ch, s] == 1)
        model.cons.add(expr=model.chat[0, a, ch, s] == 1)
        for l in model.L:
            model.cons.add(expr=model.c[l, a, ch, s] == leaves[l][ch][a][s - 1])
    return model


def add_profile_zero_constraints(model: pe.ConcreteModel, max_cn: int) -> pe.ConcreteModel:
    """Link the zero indicators to the copy numbers, so that chat = 0 <==> c = 0 and cddothat = 0 <==> cddot = 0.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets V1 and N, variables c, chat, cddot and cddothat, and ``model.cons`` already defined.
    max_cn : int
        Upper bound on copy number (big-M).

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the profile zero constraints added to ``model.cons``.
    """
    for a, ch, s in model.N:
        for i in model.V1:
            model.cons.add(model.c[i, a, ch, s] >= model.chat[i, a, ch, s])
            model.cons.add(model.c[i, a, ch, s] <= model.chat[i, a, ch, s] * max_cn)
            model.cons.add(model.cddot[i, a, ch, s] >= model.cddothat[i, a, ch, s])
            model.cons.add(model.cddot[i, a, ch, s] <= model.cddothat[i, a, ch, s] * max_cn)
    return model


def add_inheritance_constraints(model: pe.ConcreteModel) -> pe.ConcreteModel:
    """Prevent re-gaining of copies after LOH.

    A position with zero copies in the parent (chat = 0), after pre-WGD losses (cbar = 0) or after post-WGD losses
    (cddotbar = 0) stays at zero in every later stage on the edge into the child.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets E, V1 and N, variables c, chat, cbar, cdot, cddotbar and cddothat, and ``model.cons`` already
        defined.

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the inheritance constraints added to ``model.cons``.
    """
    for a, ch, s in model.N:
        for i, j in model.E:
            model.cons.add(model.chat[i, a, ch, s] >= model.cddothat[j, a, ch, s])
        for j in model.V1:
            model.cons.add(model.cddothat[j, a, ch, s] >= model.chat[j, a, ch, s])
            model.cons.add(model.cbar[j, a, ch, s] >= model.cddothat[j, a, ch, s])
            model.cons.add(model.cdot[j, a, ch, s] >= model.cbar[j, a, ch, s])
            model.cons.add(model.cddotbar[j, a, ch, s] >= model.chat[j, a, ch, s])
            model.cons.add(model.c[j, a, ch, s] >= model.cddotbar[j, a, ch, s])
    return model


def add_wgd_constraints(model: pe.ConcreteModel, max_cn: int) -> pe.ConcreteModel:
    """Add the copies duplicated by a WGD, restrict post-WGD events to WGD edges and limit WGDs among siblings.

    z = w * cdot is linearised with big-M max_cn, then cddot = cdot + z. Post-WGD gains and losses (gddot, dddot)
    and their event starts (gddotbar, dddotbar) can only be non-zero on edges with a WGD (w = 1). At most all but one
    child of an internal node can have a WGD; a node with a single child gets a bound of 1, so the constraint has no
    effect there.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets V1, N, I and E, variables w, z, cdot, cddot, gddot, dddot, gddotbar and dddotbar, and
        ``model.cons`` already defined.
    max_cn : int
        Upper bound on copy number (big-M).

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the WGD constraints added to ``model.cons``.
    """
    for j in model.V1:
        for a, ch, s in model.N:
            model.cons.add(model.z[j, a, ch, s] <= max_cn * model.w[j])
            model.cons.add(model.z[j, a, ch, s] <= model.cdot[j, a, ch, s])
            model.cons.add(model.z[j, a, ch, s] >= model.cdot[j, a, ch, s] - max_cn * (1 - model.w[j]))
            model.cons.add(model.cddot[j, a, ch, s] == model.cdot[j, a, ch, s] + model.z[j, a, ch, s])

            model.cons.add(model.gddot[j, a, ch, s] <= model.w[j] * max_cn)
            model.cons.add(model.dddot[j, a, ch, s] <= model.w[j] * max_cn)
            model.cons.add(model.gddotbar[j, a, ch, s] <= model.w[j] * max_cn)
            model.cons.add(model.dddotbar[j, a, ch, s] <= model.w[j] * max_cn)

    for i in model.I:
        i_children = [b[1] for b in model.E if b[0] == i]
        max_wgd_children = len(i_children) - 1 if len(i_children) > 1 else 1
        model.cons.add(sum(model.w[c] for c in i_children) <= max_wgd_children)
    return model


def add_start_event_pos_constraints(model: pe.ConcreteModel) -> pe.ConcreteModel:
    """Fix the event amplitudes g, d, gddot and dddot to 0 at the dummy position before each chromosome allele.

    This lets ``add_start_event_constraints`` compare segment 1 with the position before it, so an event that covers
    segment 1 counts as starting there.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets V1 and ZERO, variables g, d, gddot and dddot, and ``model.cons`` already defined.

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the start position constraints added to ``model.cons``.
    """
    for j in model.V1:
        for a, ch, s in model.ZERO:
            model.cons.add(expr=model.g[j, a, ch, s] == 0)
            model.cons.add(expr=model.d[j, a, ch, s] == 0)
            model.cons.add(expr=model.gddot[j, a, ch, s] == 0)
            model.cons.add(expr=model.dddot[j, a, ch, s] == 0)
    return model


def add_gain_and_loss_event_constraints(model: pe.ConcreteModel, max_cn: int) -> pe.ConcreteModel:
    """Chain the copy number stages on each edge through the per-position gain and loss amplitudes.

    On the edge into node j: c[parent] = cbar + d, cdot = cbar + g, cddot = cddotbar + dddot and
    c = cddotbar + gddot. The lower bound of each pair is relaxed with big-M max_cn where the later stage is zero
    (cddothat = 0 for the first two, chat = 0 for the last two), so events don't affect segments that already have
    0 copies and instead span these segments. Each amplitude is linked to its indicator (e.g. ghat = 1 <==> g > 0),
    and each position has at most one of gain / loss before the WGD and at most one after.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets E, V1 and N, the copy number and event variables (see ``spice.libs.ilp.optimal.variables``) and
        ``model.cons`` already defined.
    max_cn : int
        Upper bound on copy number and event amplitude (big-M).

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the gain and loss constraints added to ``model.cons``.
    """
    for a, ch, s in model.N:
        for i, j in model.E:
            model.cons.add(model.c[i, a, ch, s] <= model.cbar[j, a, ch, s] + model.d[j, a, ch, s])
            model.cons.add(model.c[i, a, ch, s] >= model.cbar[j, a, ch, s] + model.d[j, a, ch, s]
                           - (max_cn * (1 - model.cddothat[j, a, ch, s])))
        for j in model.V1:
            model.cons.add(model.cdot[j, a, ch, s] <= model.cbar[j, a, ch, s] + model.g[j, a, ch, s])
            model.cons.add(model.cdot[j, a, ch, s] >= model.cbar[j, a, ch, s] + model.g[j, a, ch, s]
                           - (max_cn * (1 - model.cddothat[j, a, ch, s])))
            model.cons.add(model.cddot[j, a, ch, s] <= model.cddotbar[j, a, ch, s] + model.dddot[j, a, ch, s])
            model.cons.add(model.cddot[j, a, ch, s] >= model.cddotbar[j, a, ch, s] + model.dddot[j, a, ch, s]
                           - (max_cn * (1 - model.chat[j, a, ch, s])))
            model.cons.add(model.c[j, a, ch, s] <= model.cddotbar[j, a, ch, s] + model.gddot[j, a, ch, s])
            model.cons.add(model.c[j, a, ch, s] >= model.cddotbar[j, a, ch, s] + model.gddot[j, a, ch, s]
                           - (max_cn * (1 - model.chat[j, a, ch, s])))

            model.cons.add(model.g[j, a, ch, s] >= model.ghat[j, a, ch, s])
            model.cons.add(model.g[j, a, ch, s] <= max_cn * model.ghat[j, a, ch, s])
            model.cons.add(model.d[j, a, ch, s] >= model.dhat[j, a, ch, s])
            model.cons.add(model.d[j, a, ch, s] <= max_cn * model.dhat[j, a, ch, s])
            model.cons.add(model.gddot[j, a, ch, s] >= model.gddothat[j, a, ch, s])
            model.cons.add(model.gddot[j, a, ch, s] <= max_cn * model.gddothat[j, a, ch, s])
            model.cons.add(model.dddot[j, a, ch, s] >= model.dddothat[j, a, ch, s])
            model.cons.add(model.dddot[j, a, ch, s] <= max_cn * model.dddothat[j, a, ch, s])

            model.cons.add(model.ghat[j, a, ch, s] + model.dhat[j, a, ch, s] <= 1)
            model.cons.add(model.gddothat[j, a, ch, s] + model.dddothat[j, a, ch, s] <= 1)
    return model


def add_start_event_constraints(model: pe.ConcreteModel) -> pe.ConcreteModel:
    """Count where events start: each start variable is at least the rise in its event amplitude from the previous
    position.

    gbar[j, s] >= g[j, s] - g[j, s - 1], and likewise for dbar, gddotbar and dddotbar. For segment 1 the previous
    position is the dummy position 0 (see ``add_start_event_pos_constraints``). The objective minimises the sum of
    the starts, so at the optimum each one is the number of events of that type starting at the position.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets V1 and N, variables g, d, gddot and dddot (over N0), their start variables gbar, dbar,
        gddotbar and dddotbar, and ``model.cons`` already defined.

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the event start constraints added to ``model.cons``.
    """
    for j in model.V1:
        for a, ch, s in model.N:
            model.cons.add(model.gbar[j, a, ch, s] >= model.g[j, a, ch, s] - model.g[j, a, ch, s - 1])
            model.cons.add(model.dbar[j, a, ch, s] >= model.d[j, a, ch, s] - model.d[j, a, ch, s - 1])
            model.cons.add(model.gddotbar[j, a, ch, s] >= model.gddot[j, a, ch, s] - model.gddot[j, a, ch, s - 1])
            model.cons.add(model.dddotbar[j, a, ch, s] >= model.dddot[j, a, ch, s] - model.dddot[j, a, ch, s - 1])
    return model
