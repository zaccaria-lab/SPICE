from spice.core.optimal_tree.model_variables import *
from spice.core.optimal_tree.model_constraints import *
from spice.core.optimal_tree.model_objective import *


def add_model_vars(model, leaves, edges, max_cn, allele_specific):
    n_leaves = len(leaves)
    model, n_loci = add_model_sets(model, leaves, allele_specific, edges)
    # Add variables
    model = add_copyprofile_variables(model, max_cn)
    model = add_event_variables(model, max_cn)
    model = add_start_event_variables(model, max_cn)
    model = add_wgd_variables(model)
    return model, n_loci


def add_model_cons(model, leaves, max_cn, allele_specific):
    n_leaves = len(leaves)
    model = add_input_profile_constraints(model, leaves, allele_specific, n_leaves)
    model = add_profile_zero_constraints(model, max_cn)
    model = add_inheritance_constraints(model)
    model = add_gain_and_loss_event_constraints(model, max_cn)
    model = add_wgd_constraints(model, max_cn)
    model = add_start_event_pos_constraints(model)
    model = add_start_event_constraints(model)
    return model


def add_model_obj(model, max_cn, n_loci, weight_wgd, node_weight_dict):
    model = add_objective_function(model, max_cn, n_loci, weight_wgd, node_weight_dict)
    return model
