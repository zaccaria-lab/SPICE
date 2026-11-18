# topologies

Saved as `spice_topologies.tsv.gz`.

The topologies used in the ILP, with their bootstrap support and their share of the solutions and the time limit, for debugging. One row per topology and edge. Only present with `debug=True`.

| **Column** | **Description** |
|----------|-----------------|
| `tree` | The edge, as a (parent, child) pair of SPICE's internal node IDs. |
| `topology_id` | The topology, numbered by bootstrap frequency (1 = the most frequent). Equivalent to `tree_id` in the other tables. |
| `proportion` | The topology's share of the bootstrap replicates that gave one of the topologies used. |
| `n_solutions` | The number of alternative solutions to enumerate for the topology (its share of `solution_pool_size`, at least 1). |
| `tree_tl` | The topology's time limit, in seconds: half of `time_limit` split equally between the topologies, and half in proportion to `proportion`. |
| `optimal_tl`, `multiple_tl` | The time limits of the optimal and the multiple-solutions ILP, in seconds (30% and 70% of `tree_tl`, rounded down). |
| `bootstrap_frequency` | The fraction of all bootstrap replicates that gave the topology; empty for a fixed topology. |
| `Parent`, `Child` | The edge, with the node names used in the other tables. |

[Back to the outputs](../README.md#outputs)
