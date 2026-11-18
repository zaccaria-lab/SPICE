"""Objective function for the SPICE optimal ILP.

The objective is minimised and has two parts:

fx  : the number of copy number events on the tree, i.e. the number of event starts (gbar, dbar, gddotbar, dddotbar)
      over every non-root node and position, plus a fixed cost for each WGD (``max_cn`` when ``weight_wgd`` is True,
      otherwise 1)
fx1 : a tie-breaking term, 1 - num / denom. num is the node-weighted sum of the event amplitudes (g, d, gddot, dddot)
      and WGD costs; denom is its largest possible value (every amplitude at ``max_cn`` at every position, and a WGD
      on every edge). fx1 breaks ties between solutions with the same fx. Minimising it favours larger amplitudes on
      nodes with larger weights; with the weights from ``spice.libs.ilp.optimal.solve_model.get_node_depth_weighting``,
      these are the shallower nodes, so events closer to the root are preferred.

Sets
----
V1  : non-root nodes, i.e. nodes with an incoming edge
N   : genome positions (allele, chr, seg)

Variables
---------
w                               : binary, 1 if a WGD occurs on the edge into j
gbar, dbar, gddotbar, dddotbar  : number of pre-WGD gain / loss and post-WGD gain / loss events starting at the
                                  position
g, d, gddot, dddot              : pre-WGD gain / loss and post-WGD gain / loss amplitude at each position
"""

import pyomo.environ as pe


def add_objective_function(model: pe.ConcreteModel, max_cn: int, n_loci: int, weight_wgd: bool,
                           node_weight_dict: dict) -> pe.ConcreteModel:
    """Add the objective: minimise the number of events plus the WGD cost, with a node-weighted amplitude term.

    The main term counts event starts (gbar, dbar, gddotbar, dddotbar) plus the cost of each WGD over all non-root
    nodes. The second term is a subordinate objective that breaks ties in favour of trees with events higher up the
    tree.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model with sets V1 and N and variables w, g, d, gddot, dddot, gbar, dbar, gddotbar and dddotbar already
        defined (see ``add_model_sets``, ``add_event_variables``, ``add_start_event_variables`` and
        ``add_wgd_variables`` in ``variables``).
    max_cn : int
        Maximum copy number, used as the cost of a WGD when ``weight_wgd`` is True, and to scale the amplitude term.
    n_loci : int
        Number of genome positions (size of N), from ``add_model_sets``; used to scale the amplitude term.
    weight_wgd : bool
        If True, each WGD costs ``max_cn``; if False, it costs 1.
    node_weight_dict : dict
        Weight of each non-root node in the amplitude term, {node: weight}, e.g. from ``get_node_depth_weighting``.

    Returns
    -------
    pyomo.environ.ConcreteModel
        The model with the objective Cost added.
    """
    wgd_value = max_cn if weight_wgd else 1
    node_weight = {j: 1/len(model.V1) for j in model.V1} if node_weight_dict is None else node_weight_dict

    fx = sum(model.w[j] * wgd_value + sum(model.gbar[j, a, ch, s]
                                        + model.dbar[j, a, ch, s]
                                        + model.gddotbar[j, a, ch, s]
                                        + model.dddotbar[j, a, ch, s] for a, ch, s in model.N)
             for j in model.V1)

    num = sum((model.w[j] * wgd_value + sum(model.g[j, a, ch, s]
                                          + model.d[j, a, ch, s]
                                          + model.gddot[j, a, ch, s]
                                          + model.dddot[j, a, ch, s] for a, ch, s in model.N)) * node_weight[j]
              for j in model.V1)
    denom = sum((wgd_value + (max_cn * n_loci * 4)) * node_weight[j] for j in model.V1)
    fx1 = 1 - (num/denom)

    model.Cost = pe.Objective(expr=fx + fx1, sense=pe.minimize)

    return model
