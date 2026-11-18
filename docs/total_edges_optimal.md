# total_edges_optimal

Saved as `spice_total_edges_optimal.tsv.gz`.

The edges of the optimal solution of each topology, with the events on each edge and the solution's costs. One row per topology and edge.

† Not always present; the description says when.

| **Column** | **Description** |
|----------|-----------------|
| `tree_id` | The topology. |
| `Solution` | The solution, `o<n>` for the optimal solution of topology n. |
| `Parent`, `Child` | The edge, from parent to child node (named as in [total_nodes_optimal](total_nodes_optimal.md)). |
| `Events` | The number of events on the edge (`Gains` + `Deletions`). |
| `Deletions`, `Gains` | The number of deletions and gains on the edge in the ILP; an event of amplitude 2 counts twice. |
| `WGD` | 1 if there is a whole genome duplication on the edge, otherwise 0. |
| `Total_cost` | The number of events in the solution's ILP. |
| `Gurobi_cost` | The ILP objective of the solution: `Total_cost` plus the cost of each WGD and a subordinate tie-breaking term that favours events closer to the root. |
| `Total_cost_focal` † | Only if small segments were reconstructed with the focal evolutionary model: `Total_cost` plus every event at those segments. |
| `Total_cost_focal_interval` † | Only if small segments were reconstructed with the focal evolutionary model: like `Total_cost_focal`, but without events that only extend an event already counted at a neighbouring segment. |
| `is_trunk` | True for the edge from the diploid root to the most recent common ancestor of the leaves. |

[Back to the outputs](../README.md#outputs)
