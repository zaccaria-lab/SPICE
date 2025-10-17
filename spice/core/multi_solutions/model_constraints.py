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
            model.cons.add(expr=model.c[l, a, ch, s] == leaves[l - n_leaves + 1][ch][a][s - 1])
    return model


# 2.2 For all copy number profiles, ensure that chat = 0 <==> c = 0 and ̈cddothat = 0 <==> cddot = 0.
def add_profile_zero_constraints(model, max_cn):
    for a, ch, s in model.N:
        for j in model.V1:
            model.cons.add(model.c[j, a, ch, s] >= model.chat[j, a, ch, s])
            model.cons.add(model.c[j, a, ch, s] <= model.chat[j, a, ch, s] * max_cn)
            model.cons.add(model.cddot[j, a, ch, s] >= model.cddothat[j, a, ch, s])
            model.cons.add(model.cddot[j, a, ch, s] <= model.cddothat[j, a, ch, s] * max_cn)
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


# 2.4 Define gains and losses
def add_gain_and_loss_event_constraints(model, max_cn):
    for a, ch, s in model.N:   
        for i, j in model.E:
            model.cons.add(model.c[i, a, ch, s] <= model.cbar[j, a, ch, s] + model.d[j, a, ch, s])
            model.cons.add(model.c[i, a, ch, s] >= model.cbar[j, a, ch, s] + model.d[j, a, ch, s])
        for j in model.V1:
            model.cons.add(model.cdot[j, a, ch, s] == model.cbar[j, a, ch, s] + model.g[j, a, ch, s])                                                                                   #! 75
            model.cons.add(model.cddot[j, a, ch, s] == model.cddotbar[j, a, ch, s] + model.dddot[j, a, ch, s])                                                                          #! 78, 79
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


# 2.5 Define interval (start) events
def add_interval_event_constraints(model, max_cn):
    for a, ch, s in model.N:
        use_intervals = [(p, q) for a1, ch1, p, q in model.PQ if (p <= s <= q) & (ch1 == ch) & (a1 == a)]
        for j in model.V1:
            model.cons.add(model.g[j, a, ch, s] == pe.quicksum(model.gbar[j, a, ch, p, q] for p, q in use_intervals))
            model.cons.add(model.d[j, a, ch, s] == pe.quicksum(model.dbar[j, a, ch, p, q] for p, q in use_intervals))
            model.cons.add(model.gddot[j, a, ch, s] == pe.quicksum(model.gddotbar[j, a, ch, p, q] for p, q in use_intervals))
            model.cons.add(model.dddot[j, a, ch, s] == pe.quicksum(model.dddotbar[j, a, ch, p, q] for p, q in use_intervals))      
    model.igbar1 = pe.Constraint(model.V1, model.PQ, rule=(lambda model, j, a, ch, p, q: model.gbar[j, a, ch, p, q] >= model.gbarhat[j, a, ch, p, q]))
    model.igbar2 = pe.Constraint(model.V1, model.PQ, rule=(lambda model, j, a, ch, p, q: model.gbar[j, a, ch, p, q] <= model.gbarhat[j, a, ch, p, q] * max_cn))
    model.idbar1 = pe.Constraint(model.V1, model.PQ, rule=(lambda model, j, a, ch, p, q: model.dbar[j, a, ch, p, q] >= model.dbarhat[j, a, ch, p, q]))
    model.idbar2 = pe.Constraint(model.V1, model.PQ, rule=(lambda model, j, a, ch, p, q: model.dbar[j, a, ch, p, q] <= model.dbarhat[j, a, ch, p, q] * max_cn))
    model.igddotbar1 = pe.Constraint(model.V1, model.PQ, rule=(lambda model, j, a, ch, p, q: model.gddotbar[j, a, ch, p, q] >= model.gddotbarhat[j, a, ch, p, q]))
    model.igddotbar2 = pe.Constraint(model.V1, model.PQ, rule=(lambda model, j, a, ch, p, q: model.gddotbar[j, a, ch, p, q] <= model.gddotbarhat[j, a, ch, p, q] * max_cn))
    model.idddotbar1 = pe.Constraint(model.V1, model.PQ, rule=(lambda model, j, a, ch, p, q: model.dddotbar[j, a, ch, p, q] >= model.dddotbarhat[j, a, ch, p, q]))
    model.idddotbar2 = pe.Constraint(model.V1, model.PQ, rule=(lambda model, j, a, ch, p, q: model.dddotbar[j, a, ch, p, q] <= model.dddotbarhat[j, a, ch, p, q] * max_cn))
    
    
    model.ogbar = pe.Constraint(model.V1, model.O, rule=(lambda model, j, a, ch1, p1, q1, p2, q2: model.gbarhat[j, a, ch1, p1, q1] + model.gbarhat[j, a, ch1, p2, q2] <= 1))
    model.odbar = pe.Constraint(model.V1, model.O, rule=(lambda model, j, a, ch1, p1, q1, p2, q2: model.dbarhat[j, a, ch1, p1, q1] + model.dbarhat[j, a, ch1, p2, q2] <= 1))
    model.ogddotbar = pe.Constraint(model.V1, model.O, rule=(lambda model, j, a, ch1, p1, q1, p2, q2: model.gddotbarhat[j, a, ch1, p1, q1] + model.gddotbarhat[j, a, ch1, p2, q2] <= 1))
    model.odddotbar = pe.Constraint(model.V1, model.O, rule=(lambda model, j, a, ch1, p1, q1, p2, q2: model.dddotbarhat[j, a, ch1, p1, q1] + model.dddotbarhat[j, a, ch1, p2, q2] <= 1))
    return model   
    
def add_cherry_constraints(model, max_cn):
    for i in model.I:
        i_children = [b[1] for b in model.E if b[0] == i]
        max_cherry_children = len(i_children) - 1 if len(i_children) > 1 else 1 
        for a, ch, p, q in model.PQ:
            model.cons.add(sum(model.gbarhat[c, a, ch, p, q] for c in i_children) <= max_cherry_children)
            model.cons.add(sum(model.dbarhat[c, a, ch, p, q] for c in i_children)  <= max_cherry_children)
            model.cons.add(sum(model.gddotbarhat[c, a, ch, p, q] for c in i_children) <= max_cherry_children)
            model.cons.add(sum(model.dddotbarhat[c, a, ch, p, q] for c in i_children) <= max_cherry_children)
    return model
            

# 2.6 Define WGD events 
def add_wgd_constraints(model, max_cn):
    for j in model.V1:
        for a, ch, s in model.N:
            model.cons.add(model.z[j, a, ch, s] <= max_cn * model.w[j])
            model.cons.add(model.z[j, a, ch, s] <= model.cdot[j, a, ch, s]) 
            model.cons.add(model.z[j, a, ch, s] >= model.cdot[j, a, ch, s] - max_cn * (1 - model.w[j]))
            model.cons.add(model.cddot[j, a, ch, s] == model.cdot[j, a, ch, s] + model.z[j, a, ch, s]) 
            model.cons.add(model.gddot[j, a, ch, s] <= model.w[j] * max_cn) 
            model.cons.add(model.dddot[j, a, ch, s] <= model.w[j] * max_cn)          
    return model
                
    
