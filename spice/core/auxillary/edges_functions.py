import numpy as np
import math

def get_fixed_edges(leaves, rs=None):
    leaves = list(leaves.keys())
    nodes, edges = make_tree(leaves, rs)
    label_dict = {i: len(leaves) - 1 + i if i <= len(leaves) else i - len(leaves) for i in range(1, 2*len(leaves))}
    return [(0, 1)] + [(label_dict[i], label_dict[j]) for i, j in edges]


def make_tree(leaves, random_seed):
    parents = [max(leaves)] 
    tree_edges, parents, current_parent = make_tree_internal(leaves, parents, random_seed)
    tree_nodes = leaves + parents[1:] 
    return tree_nodes, tree_edges


def make_tree_internal(leaves, parents, random_seed):
    if len(leaves) == 1:
        return [], parents, leaves[0]     
    else:
        current_parent = max(parents) + 1 
        parents.append(current_parent)
        left_leaves, right_leaves = split(leaves, random_seed)
        tree, parents, parent_left = make_tree_internal(left_leaves, parents, random_seed)
        tree_right, parents, parent_right = make_tree_internal(right_leaves, parents, random_seed)
        tree.extend(tree_right)
        tree.extend([(current_parent, parent_left), (current_parent, parent_right)])
        return tree, parents, current_parent


def split(leaves, random_seed): 
    leaves = np.array(leaves)
    n_leaves = len(leaves)
    n_ones = np.random.randint(1, n_leaves)
    n_zeros = n_leaves - n_ones
    mask = np.array(n_ones * [1] + n_zeros * [0], dtype=bool)
    np.random.seed(random_seed)
    np.random.shuffle(mask)
    left_leaves = leaves[mask].tolist()
    right_leaves = leaves[~mask].tolist()
    return left_leaves, right_leaves


def get_edges(nleaves):
    return list(enumerate_binconsensus(nleaves).union({(i, j) for i in range(1, nleaves) for j in range(nleaves, 2*nleaves)}))

def enumerate_binconsensus(nleaves):
    return {edge for edge in enumerate_innerconsensus(0, np.arange(1, nleaves), np.arange(nleaves, 2 * nleaves))}

def enumerate_innerconsensus(parent, internals, leaves):
    if len(leaves) == 1:
        yield (parent, leaves[0])
    else:
        root = min(internals)
        yield (parent, root)
        for split in range(1, math.floor(leaves.size / 2.0) + 1):
            yield from enumerate_innerconsensus(root, internals[1:split], leaves[:split])
            yield from enumerate_innerconsensus(root, internals[split:], leaves[split:])