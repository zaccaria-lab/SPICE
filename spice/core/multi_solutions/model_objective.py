import numpy as np
import pyomo.environ as pe

def add_objective_function(model, max_cn, weight_wgd):
    wgd_value = max_cn if weight_wgd else 1    
    fx = sum((model.w[j]*wgd_value) + sum(model.gbar[j, a, ch, p, q] 
                                        + model.dbar[j, a, ch, p, q] 
                                        + model.gddotbar[j, a, ch, p, q] 
                                        + model.dddotbar[j, a, ch, p, q] for a, ch, p, q in model.PQ) 
                                                                         for j in model.V1) 
    model.Cost = pe.Objective(expr=fx, sense=pe.minimize)
    return model  