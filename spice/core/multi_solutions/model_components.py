from spice.core.multi_solutions.model_variables import *
from spice.core.multi_solutions.model_constraints import *
from spice.core.multi_solutions.model_objective import *

def add_model_vars(model, leaves, edges, max_cn, allele_specific, init_model=None, init_model_vars=[], fix_wgd=True, logger=None):
    n_leaves = len(leaves)
    model, n_loci, extra_segs = add_model_sets(model, leaves, allele_specific, edges, logger)
    # Add variables
    model = add_copyprofile_variables(model, max_cn, init_model, init_model_vars)
    model = add_event_variables(model, max_cn, init_model, init_model_vars)
    model = add_interval_event_variables(model, max_cn)
    model, init_w = add_wgd_variables(model, init_model, init_model_vars)
    return model, n_loci, init_w, extra_segs


def add_model_cons(model, leaves, max_cn, allele_specific):
    n_leaves = len(leaves)
    model = add_input_profile_constraints(model, leaves, allele_specific, n_leaves)
    model = add_profile_zero_constraints(model, max_cn)
    model = add_inheritance_constraints(model)
    model = add_gain_and_loss_event_constraints(model, max_cn)
    model = add_wgd_constraints(model, max_cn)
    model = add_interval_event_constraints(model, max_cn)
    model = add_cherry_constraints(model, max_cn)
    return model


def add_model_obj(model, max_cn, weight_wgd):
    model = add_objective_function(model, max_cn, weight_wgd)
    return model

ensure_initialize_integers = (lambda model_var, keys : {k : round(v) if v is not None else v for k, v in model_var.items() if k in keys})