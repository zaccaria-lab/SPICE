# total_debug

Saved as `spice_debug.tsv.gz` (with the variable names as the first column; read it back with `pd.read_csv(..., sep='\t', index_col=0)`).

The value of every ILP variable in each solution, for debugging. One row per variable and index, one column per solution. Only present with `debug=True`.

| **Column** | **Description** |
|----------|-----------------|
| index | The ILP variable and its index, e.g. `c(0_0_1_1)`: the variable name, then the node, allele (`0` = A, `1` = B), chromosome and ILP segment, or for interval variables (e.g. `gbar(1_0_4_1_3)`) the node, allele, chromosome and first and last ILP segment. Nodes and segments are SPICE's internal numbering (see [mapping](mapping.md) for the segments). |
| `o<n>`, `m<n>` | The value of the variable in each solution (see [total_nodes_optimal](total_nodes_optimal.md) and [total_nodes_multisol](total_nodes_multisol.md)); empty where the solution's ILP has no such variable. |

The variables are described in the module docstrings of `spice/libs/ilp/optimal/variables.py` and `spice/libs/ilp/multi/variables.py`.

[Back to the outputs](../README.md#outputs)
