# solve_info

Saved as `spice_solve_info.tsv`.

Information on the ILP solves of each topology and on the run, as parameter / value pairs. One row per parameter and topology, plus two rows for the whole run.

| **Column** | **Description** |
|----------|-----------------|
| `parameter` | The parameter (see below). |
| `value` | Its value. |
| `tree_id` | The topology; empty for `total_run_time` and `random_seed`. |

| **Parameter** | **Description** |
|----------|-----------------|
| `optimal_tree_solve_time` | The solve time of the optimal ILP, in seconds. |
| `optimal_tree_model_status` | The Gurobi status of the optimal ILP: `optimal`, `infeasible`, `node_limit`, `time_limit`, `solution_limit` or `user_interrupt`. |
| `optimal_tree_gap` | The optimality gap of the optimal ILP, as a percentage of the upper bound (0 when solved to optimality). |
| `optimal_tree_total_cost` | The ILP objective of the optimal solution (its `Gurobi_cost` in [total_edges_optimal](total_edges_optimal.md)). |
| `multi_solution_solve_time`, `multi_solution_model_status`, `multi_solution_gap` | The same for the multiple-solutions ILP. Only with `multiple_solutions`. |
| `multi_solution_total_cost_range` | The ILP objective of the best and the worst solution found, as `best, worst`. Only with `multiple_solutions`. |
| `n_solutions` | The number of solutions found for the topology. Only with `multiple_solutions`. |
| `n_optimal_solutions` | The number of those solutions with the lowest cost. Only with `multiple_solutions`. |
| `total_run_time` | The run time of SPICE, in seconds. |
| `random_seed` | The random seed (`rs`). |

[Back to the outputs](../README.md#outputs)
