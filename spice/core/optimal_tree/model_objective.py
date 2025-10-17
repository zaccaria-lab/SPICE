import numpy as np
import pyomo.environ as pe

def add_objective_function(model, max_cn, n_loci, weight_wgd, node_weight_dict=None):
    wgd_value = max_cn if weight_wgd else 1
    if node_weight_dict is None:
        print('WARNING: Node weights are not provided. Defaulting to equal weights for all nodes.')
    node_weight = {j: 1/len(model.V1) for j in model.V1} if node_weight_dict is None else node_weight_dict
    
    fx = sum((model.w[j] * wgd_value) + sum(model.gbar[j, a, ch, s] 
                                          + model.dbar[j, a, ch, s] 
                                          + model.gddotbar[j, a, ch, s] 
                                          + model.dddotbar[j, a, ch, s] for a, ch, s in model.N) 
                                                                        for j in model.V1) 
     
    num = sum((((model.w[j] * wgd_value) + sum(model.g[j, a, ch, s]
                                        + model.d[j, a, ch, s] 
                                        + model.gddot[j, a, ch, s] 
                                        + model.dddot[j, a, ch, s] for a, ch, s in model.N)) * node_weight[j]) for j in model.V1)  
    denom = sum((wgd_value + (max_cn * n_loci * 4)) * node_weight[j] for j in model.V1)
    fx1 = 1 - (num/denom)
                            
    model.Cost = pe.Objective(expr=fx + fx1, sense=pe.minimize)

    return model  

    
        
