"""Pyomo constraints for the SPICE multiple-solutions (enumeration) ILP.

Each function adds one group of constraints to a model that already has the sets and variables from
``spice.libs.ilp.multi.variables`` and a ``ConstraintList`` at ``model.cons``.

Sets
----
V1   : non-root nodes, i.e. nodes with an incoming edge (the root is node 0)
L    : leaf nodes (observed copy number profiles)
I    : internal non-root nodes
E    : tree edges (parent, child)
N    : genome positions (allele, chr, seg), with seg 1-based
PQ   : genomic intervals (allele, chr, p, q), p <= q, on which copy number events act
O    : pairs of intervals on the same allele and chromosome that overlap or abut (allele, chr, p1, q1, p2, q2)

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
w : 1 if a WGD occurs on the edge into j
z : linearisation of w[j] * cdot[j], i.e. the copies added by the WGD

Constraint groups
-----------------
input profile   : root is diploid (1 per allele, or 2 if not allele-specific); leaves equal the observed profiles
profile zero    : chat / cddothat are 1 exactly when c / cddot is non-zero
inheritance     : prevent re-gaining of copies after LOH
gain and loss   : chains parent -> cbar -> cdot -> cddot -> cddotbar -> c through the event amplitudes; each
                  position has at most one of gain / loss before the WGD and at most one after
interval events : each per-position amplitude is the sum of the interval events covering it; overlapping or abutting
                  intervals of the same event type cannot both occur on one edge
cherry          : the same interval event cannot occur on every child edge of an internal node
WGD             : z = w * cdot (linearised), cddot = cdot + z; post-WGD events only occur on WGD edges
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
        for j in model.V1:
            model.cons.add(model.c[j, a, ch, s] >= model.chat[j, a, ch, s])
            model.cons.add(model.c[j, a, ch, s] <= model.chat[j, a, ch, s] * max_cn)
            model.cons.add(model.cddot[j, a, ch, s] >= model.cddothat[j, a, ch, s])
            model.cons.add(model.cddot[j, a, ch, s] <= model.cddothat[j, a, ch, s] * max_cn)
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


def add_gain_and_loss_event_constraints(model: pe.ConcreteModel, max_cn: int) -> pe.ConcreteModel:
    """Chain the copy number stages on each edge through the per-position gain and loss amplitudes.

    On the edge into node j: c[parent] = cbar + d, cdot = cbar + g, cddot = cddotbar + dddot and
    c = cddotbar + gddot. Each amplitude is linked to its indicator (e.g. ghat = 1 <==> g > 0), and each position
    has at most one of gain / loss before the WGD and at most one after.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets E, V1 and N, the copy number and event variables (see ``spice.libs.ilp.multi.variables``) and
        ``model.cons`` already defined.
    max_cn : int
        Upper bound on event amplitude (big-M).

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the gain and loss constraints added to ``model.cons``.
    """
    for a, ch, s in model.N:
        for i, j in model.E:
            model.cons.add(model.c[i, a, ch, s] <= model.cbar[j, a, ch, s] + model.d[j, a, ch, s])
            model.cons.add(model.c[i, a, ch, s] >= model.cbar[j, a, ch, s] + model.d[j, a, ch, s])
        for j in model.V1:
            model.cons.add(model.cdot[j, a, ch, s] == model.cbar[j, a, ch, s] + model.g[j, a, ch, s])
            model.cons.add(model.cddot[j, a, ch, s] == model.cddotbar[j, a, ch, s] + model.dddot[j, a, ch, s])
            model.cons.add(model.c[j, a, ch, s] == model.cddotbar[j, a, ch, s] + model.gddot[j, a, ch, s])

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


def add_interval_event_constraints(model: pe.ConcreteModel, max_cn: int) -> pe.ConcreteModel:
    """Build the per-position event amplitudes from interval events.

    Each per-position amplitude (g, d, gddot, dddot) equals the sum of the interval events of that type covering the
    position. Interval amplitudes are linked to their indicators (e.g. gbarhat = 1 <==> gbar > 0), and overlapping or
    abutting intervals (set O) of the same event type cannot both occur on one edge.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets V1, N, PQ and O, the event and interval event variables and ``model.cons`` already defined.
    max_cn : int
        Upper bound on interval event amplitude (big-M).

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the sum constraints added to ``model.cons``, plus the constraint components igbar1, igbar2,
        idbar1, idbar2, igddotbar1, igddotbar2, idddotbar1, idddotbar2 (indicator links) and ogbar, odbar,
        ogddotbar, odddotbar (overlapping intervals).
    """
    for a, ch, s in model.N:
        use_intervals = [(p, q) for a1, ch1, p, q in model.PQ if (p <= s <= q) and (ch1 == ch) and (a1 == a)]
        for j in model.V1:
            model.cons.add(model.g[j, a, ch, s] == pe.quicksum(model.gbar[j, a, ch, p, q] for p, q in use_intervals))
            model.cons.add(model.d[j, a, ch, s] == pe.quicksum(model.dbar[j, a, ch, p, q] for p, q in use_intervals))
            model.cons.add(model.gddot[j, a, ch, s] ==
                           pe.quicksum(model.gddotbar[j, a, ch, p, q] for p, q in use_intervals))
            model.cons.add(model.dddot[j, a, ch, s] ==
                           pe.quicksum(model.dddotbar[j, a, ch, p, q] for p, q in use_intervals))
    model.igbar1 = pe.Constraint(model.V1, model.PQ, rule=lambda model, j, a, ch, p, q:
                                 model.gbar[j, a, ch, p, q] >= model.gbarhat[j, a, ch, p, q])
    model.igbar2 = pe.Constraint(model.V1, model.PQ, rule=lambda model, j, a, ch, p, q:
                                 model.gbar[j, a, ch, p, q] <= model.gbarhat[j, a, ch, p, q] * max_cn)
    model.idbar1 = pe.Constraint(model.V1, model.PQ, rule=lambda model, j, a, ch, p, q:
                                 model.dbar[j, a, ch, p, q] >= model.dbarhat[j, a, ch, p, q])
    model.idbar2 = pe.Constraint(model.V1, model.PQ, rule=lambda model, j, a, ch, p, q:
                                 model.dbar[j, a, ch, p, q] <= model.dbarhat[j, a, ch, p, q] * max_cn)
    model.igddotbar1 = pe.Constraint(model.V1, model.PQ, rule=lambda model, j, a, ch, p, q:
                                     model.gddotbar[j, a, ch, p, q] >= model.gddotbarhat[j, a, ch, p, q])
    model.igddotbar2 = pe.Constraint(model.V1, model.PQ, rule=lambda model, j, a, ch, p, q:
                                     model.gddotbar[j, a, ch, p, q] <= model.gddotbarhat[j, a, ch, p, q] * max_cn)
    model.idddotbar1 = pe.Constraint(model.V1, model.PQ, rule=lambda model, j, a, ch, p, q:
                                     model.dddotbar[j, a, ch, p, q] >= model.dddotbarhat[j, a, ch, p, q])
    model.idddotbar2 = pe.Constraint(model.V1, model.PQ, rule=lambda model, j, a, ch, p, q:
                                     model.dddotbar[j, a, ch, p, q] <= model.dddotbarhat[j, a, ch, p, q] * max_cn)

    model.ogbar = pe.Constraint(model.V1, model.O, rule=lambda model, j, a, ch1, p1, q1, p2, q2:
                                model.gbarhat[j, a, ch1, p1, q1] + model.gbarhat[j, a, ch1, p2, q2] <= 1)
    model.odbar = pe.Constraint(model.V1, model.O, rule=lambda model, j, a, ch1, p1, q1, p2, q2:
                                model.dbarhat[j, a, ch1, p1, q1] + model.dbarhat[j, a, ch1, p2, q2] <= 1)
    model.ogddotbar = pe.Constraint(model.V1, model.O, rule=lambda model, j, a, ch1, p1, q1, p2, q2:
                                    model.gddotbarhat[j, a, ch1, p1, q1] + model.gddotbarhat[j, a, ch1, p2, q2] <= 1)
    model.odddotbar = pe.Constraint(model.V1, model.O, rule=lambda model, j, a, ch1, p1, q1, p2, q2:
                                    model.dddotbarhat[j, a, ch1, p1, q1] + model.dddotbarhat[j, a, ch1, p2, q2] <= 1)
    return model


def add_cherry_constraints(model: pe.ConcreteModel) -> pe.ConcreteModel:
    """Stop the same interval event occurring on every child edge of an internal node.

    Such an event would belong on the edge into the node instead. A node with a single child gets a bound of 1, so
    the constraint has no effect there.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets I, E and PQ, the interval event indicators (*barhat) and ``model.cons`` already defined.

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the cherry constraints added to ``model.cons``.
    """
    for i in model.I:
        i_children = [b[1] for b in model.E if b[0] == i]
        max_cherry_children = len(i_children) - 1 if len(i_children) > 1 else 1
        for a, ch, p, q in model.PQ:
            model.cons.add(sum(model.gbarhat[c, a, ch, p, q] for c in i_children) <= max_cherry_children)
            model.cons.add(sum(model.dbarhat[c, a, ch, p, q] for c in i_children) <= max_cherry_children)
            model.cons.add(sum(model.gddotbarhat[c, a, ch, p, q] for c in i_children) <= max_cherry_children)
            model.cons.add(sum(model.dddotbarhat[c, a, ch, p, q] for c in i_children) <= max_cherry_children)
    return model


def add_wgd_constraints(model: pe.ConcreteModel, max_cn: int) -> pe.ConcreteModel:
    """Add the copies duplicated by a WGD and restrict post-WGD events to WGD edges.

    z = w * cdot is linearised with big-M max_cn, then cddot = cdot + z. Post-WGD gains and losses (gddot, dddot)
    can only be non-zero on edges with a WGD (w = 1).

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets V1 and N, variables w, z, cdot, cddot, gddot and dddot, and ``model.cons`` already defined.
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
    return model
