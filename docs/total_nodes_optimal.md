# total_nodes_optimal

Saved as `spice_total_nodes_optimal.tsv.gz`.

The copy numbers of the optimal solution of each topology, which is used to warm start the enumeration of alternative solutions. One row per topology, node and merged segment (consecutive input segments with the same copy numbers in every leaf).

† Not always present; the description says when.

| **Column** | **Description** |
|----------|-----------------|
| `tree_id` | The topology. |
| `Solution` | The solution, `o<n>` for the optimal solution of topology n. |
| `Node` | The node. Internal nodes are numbered, with `0` the diploid root; leaves keep their input names, except that a name of digits becomes `clone_<name>` if the topology was inferred. |
| `Chr`, `Start`, `End` | The merged segment. |
| `cnA`, `cnB` † | Allele-specific mode: the copy number of alleles A and B. |
| `CN` † | Total copy number mode: the total copy number. |
| `DeletionsA`, `DeletionsB`, `GainsA`, `GainsB` † | Allele-specific mode: the amplitude of the deletions and gains of each allele at this segment on the branch into the node. |
| `Deletions`, `Gains` † | Total copy number mode: the amplitude of the deletions and gains at this segment on the branch into the node. |
| `cn_model` † | Only if small segments were reconstructed with the focal evolutionary model: the model that inferred the copy numbers, `ilp_model` or `focal_model`. |

[Back to the outputs](../README.md#outputs)
