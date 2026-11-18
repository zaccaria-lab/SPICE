# total_events

Saved as `spice_total_events.tsv.gz`.

The copy number events of the alternative enumerated solutions. One row per event, or per piece of an event that was split (see `event_id`).

† Not always present; the description says when.

| **Column** | **Description** |
|----------|-----------------|
| `tree_id` | The topology. |
| `Solution` | The solution, `m<n>` (see [total_nodes_multisol](total_nodes_multisol.md)). |
| `Node` | The node at the end of the branch with the event. |
| `Chr`, `Start`, `End` | The region the event covers. |
| `Allele` † | Allele-specific mode: the allele, `A` or `B`. |
| `Event` | The event type: `Gain` or `Deletion`, or `Gain_postWGD` / `Deletion_postWGD` for an ILP event after a WGD on the same branch. |
| `nEvents` | The amplitude of the event. |
| `cn_model` † | Only if small segments were reconstructed with the focal evolutionary model: the model that inferred the event, `ilp_model` (an interval event) or `focal_model` (an event at one small segment). |
| `event_id` | Identifies the event within a solution, and the same event across solutions: `<model>_<node>.<chr>.<allele>.<event>.<Start>_<End>`, e.g. `ilp_1.1.A.Gain.1000000_5000000`, or without the allele in total copy number mode, e.g. `ilp_1.1.Gain.1000000_5000000`. An ILP event that spans small segments is split into one row per run of ILP segments; the rows share the ID, which has the coordinates before splitting. |

Summing `nEvents` over the unique `event_id`s of a solution gives its `Total_cost` or, if small segments were reconstructed with the focal evolutionary model, its `Total_cost_focal` (see [total_edges_multisol](total_edges_multisol.md)).

[Back to the outputs](../README.md#outputs)
