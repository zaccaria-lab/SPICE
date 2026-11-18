# mapping

Saved as `spice_mapping.tsv.gz`.

How the input segments map to the segments SPICE works on internally, for debugging. One row per input segment and allele. Only present with `debug=True`.

| **Column** | **Description** |
|----------|-----------------|
| `Chr`, `Start`, `End` | The input segment. |
| `Allele` | The allele, `A` or `B` (always `A` in total copy number mode). |
| `origPos` | The input segment's number, from 1 on each chromosome. |
| `mergePos` | The merged segment it belongs to (consecutive input segments with the same copy numbers in every leaf), as in the node tables. |
| `filteredPos` | The segment's number after removing the small segments; empty for small segments (reconstructed with the focal evolutionary model, or removed with `rm_small_segs`). |
| `finalPos` | The ILP segment: consecutive segments with the same copy numbers in every leaf after filtering, merged again; empty for small segments. Numbered as the segments in [total_debug](total_debug.md). |

[Back to the outputs](../README.md#outputs)
