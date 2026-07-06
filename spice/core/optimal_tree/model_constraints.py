import pyomo.environ as pe
import numpy as np

# 2.1 Define known copy-number profiles
def add_input_profile_constraints(model, leaves, allele_specific, n_leaves):
    for a, ch, s in model.N:
        if allele_specific == False:
            model.cons.add(expr=model.c[0, a, ch, s] == 2)
        else:
            model.cons.add(expr=model.c[0, a, ch, s] == 1)
        model.cons.add(expr=model.chat[0, a, ch, s] == 1)
        for l in model.L:
            model.cons.add(expr=model.c[l, a, ch, s] == leaves[l][ch][a][s - 1])
    return model


# 2.2 For all copy number profiles, ensure that chat = 0 <==> c = 0 and ̈cddothat = 0 <==> cddot = 0.
def add_profile_zero_constraints(model, max_cn):
    for a, ch, s in model.N:
        for i in model.V1:
            model.cons.add(model.c[i, a, ch, s] >= model.chat[i, a, ch, s])
            model.cons.add(model.c[i, a, ch, s] <= model.chat[i, a, ch, s] * max_cn)
            model.cons.add(model.cddot[i, a, ch, s] >= model.cddothat[i, a, ch, s])
            model.cons.add(model.cddot[i, a, ch, s] <= model.cddothat[i, a, ch, s] * max_cn)
    return model


# 2.3 Define the inheritance of copy-numbers from parent to child
def add_inheritance_constraints(model):   
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


# 2.4 Define WGD events
def add_wgd_constraints(model, max_cn):
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


# 2.5 Defining gains and losses
def add_start_event_pos_constraints(model):
    for j in model.V1:
        for a, ch, s in model.ZERO:
            model.cons.add(expr=model.g[j, a, ch, s] == 0)
            model.cons.add(expr=model.d[j, a, ch, s] == 0)
            model.cons.add(expr=model.gddot[j, a, ch, s] == 0)
            model.cons.add(expr=model.dddot[j, a, ch, s] == 0)
    return model

def add_gain_and_loss_event_constraints(model, max_cn):
    for a, ch, s in model.N:
        for i, j in model.E: 
            model.cons.add(model.c[i, a, ch, s] <= model.cbar[j, a, ch, s] + model.d[j, a, ch, s])
            model.cons.add(model.c[i, a, ch, s] >= model.cbar[j, a, ch, s] + model.d[j, a, ch, s] - (max_cn * (1 - model.cddothat[j, a, ch, s])))
        for j in model.V1:
            model.cons.add(model.cdot[j, a, ch, s] <= model.cbar[j, a, ch, s] + model.g[j, a, ch, s]) 
            model.cons.add(model.cdot[j, a, ch, s] >= model.cbar[j, a, ch, s] + model.g[j, a, ch, s] - (max_cn * (1 - model.cddothat[j, a, ch, s]))) 
            model.cons.add(model.cddot[j, a, ch, s] <= model.cddotbar[j, a, ch, s] + model.dddot[j, a, ch, s]) 
            model.cons.add(model.cddot[j, a, ch, s] >= model.cddotbar[j, a, ch, s] + model.dddot[j, a, ch, s] - (max_cn * (1 - model.chat[j, a, ch, s])))
            model.cons.add(model.c[j, a, ch, s] <= model.cddotbar[j, a, ch, s] + model.gddot[j, a, ch, s])
            model.cons.add(model.c[j, a, ch, s] >= model.cddotbar[j, a, ch, s] + model.gddot[j, a, ch, s] - (max_cn * (1 - model.chat[j, a, ch, s]))) 
            
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


# 2.6 Define interval (start) events
def add_start_event_constraints(model):
    for j in model.V1:
        for a, ch, s in model.N:
            model.cons.add(model.gbar[j, a, ch, s] >= model.g[j, a, ch, s] - model.g[j, a, ch, s-1])
            model.cons.add(model.dbar[j, a, ch, s] >= model.d[j, a, ch, s] - model.d[j, a, ch, s-1])
            model.cons.add(model.gddotbar[j, a, ch, s] >= model.gddot[j, a, ch, s] - model.gddot[j, a, ch, s-1])
            model.cons.add(model.dddotbar[j, a, ch, s] >= model.dddot[j, a, ch, s] - model.dddot[j, a, ch, s-1])
    return model




