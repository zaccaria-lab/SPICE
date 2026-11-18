# cn_probabilities

Saved as `spice_cn_probabilities.tsv.gz`.

The probability of each copy number of each internal node, computed separately for each topology from its alternative enumerated solutions. One row per topology, internal node, merged segment, allele and copy number; only copy numbers found in at least one solution are listed. Only present if `rank_multiple_trees` (and `multiple_solutions`) is True.

† Not always present; the description says when.

| **Column** | **Description** |
|----------|-----------------|
| `tree_id` | The topology. |
| `top_ranked_topology` | True for the most probable topology (`topology_rank` 1 in [solution_probabilities](solution_probabilities.md)); if several are tied, all of them. |
| `top_ranked_solution` | True for the topology of the top-ranked solution (`solution_rank` 1); if several are tied, all of them. |
| `Node` | The internal node (the diploid root `0` and the leaves are not listed). |
| `Chr`, `Start`, `End` | The merged segment. |
| `Allele` † | Allele-specific mode: the allele, `A` or `B`. |
| `CN` | The copy number. |
| `CN_probability` | The probability of this copy number at this node and segment, given the topology: the weighted fraction of the topology's solutions with this copy number. The probabilities of a node, segment and allele sum to 1. |

[Back to the outputs](../README.md#outputs)
