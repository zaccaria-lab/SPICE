"""Objective function for the SPICE multiple-solutions (enumeration) ILP.

The objective minimises the total cost of the copy number events on the tree: the summed amplitude of every interval
gain and loss, before and after WGD, plus a fixed cost for each WGD.

Sets
----
V1   : non-root nodes, i.e. nodes with an incoming edge
PQ   : genomic intervals (allele, chr, p, q), p <= q, on which copy number events act

Variables
---------
w                   : 1 if a WGD occurs on the edge into j (Param when fixed, otherwise Var)
gbar, dbar          : amplitude of a pre-WGD interval gain / loss over PQ
gddotbar, dddotbar  : amplitude of a post-WGD interval gain / loss over PQ
"""

import pyomo.environ as pe


def add_objective_function(model: pe.ConcreteModel, max_cn: int, weight_wgd: bool) -> pe.ConcreteModel:
    """Add the objective: minimise the total event amplitude plus the WGD cost.

    The cost of each WGD is ``max_cn`` when ``weight_wgd`` is True, otherwise 1.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets V1 and PQ and variables w, gbar, dbar, gddotbar and dddotbar already defined (see
        ``add_model_sets``, ``add_interval_event_variables`` and ``add_wgd_variables``).
    max_cn : int
        Maximum copy number, used as the cost of a WGD when ``weight_wgd`` is True.
    weight_wgd : bool
        If True, each WGD costs ``max_cn``; if False, it costs 1.

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the objective Cost added.
    """
    wgd_value = max_cn if weight_wgd else 1
    fx = sum(model.w[j] * wgd_value
             + sum(model.gbar[j, a, ch, p, q]
                   + model.dbar[j, a, ch, p, q]
                   + model.gddotbar[j, a, ch, p, q]
                   + model.dddotbar[j, a, ch, p, q] for a, ch, p, q in model.PQ)
             for j in model.V1)
    model.Cost = pe.Objective(expr=fx, sense=pe.minimize)
    return model
