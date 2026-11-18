# solution_probabilities

Saved as `spice_solutions_ranked.tsv.gz`.

The probabilities and rankings of the alternative enumerated solutions. One row per solution, sorted by `solution_rank`. Only present if `rank_multiple_trees` (and `multiple_solutions`) is True.

Each solution gets a geometric weight from its cost, so that the most parsimonious solutions dominate. Treating the weights as a distribution over solutions, SPICE computes the probability of each solution's topology and of its copy numbers.

| **Column** | **Description** |
|----------|-----------------|
| `tree_id` | The topology. |
| `Solution` | The solution, `m<n>` (see [total_nodes_multisol](total_nodes_multisol.md)). |
| `solution_rank` | The rank of the solution by `solution_log_prob`: 1 is the most probable; tied solutions share a rank. |
| `topology_rank` | The rank of each topology by `topology_log_prob`, ranked like `solution_rank`. |
| `solution_log_prob` | `topology_log_prob` + `cn_log_prob`. |
| `topology_log_prob` | The log probability of the topology: for each branch, the weighted fraction of solutions in which the child node has the same ancestors, among the solutions that contain those ancestors, summed as logs over the branches. The same for every solution of a topology. |
| `cn_log_prob` | The log probability of the copy numbers: for each internal node and merged segment, the weighted fraction of solutions in which the node has the same copy number, among those with the same ancestors and ancestral copy numbers, summed as logs. |
| `geometric_weight` | The weight of the solution: the geometric probability `geom.pmf(k, p=0.8)`, with `k` the solution's `Total_cost` minus the lowest `Total_cost` plus 1. |
| `Total_cost`, `Gurobi_cost` | The solution's costs (see [total_edges_multisol](total_edges_multisol.md)). |
| `bootstrap_frequency` | The fraction of the bootstrap replicates that gave the topology; empty for a fixed topology. |
| `Tree` | The topology, as a tuple of (parent, child) edges. |

[Back to the outputs](../README.md#outputs)
