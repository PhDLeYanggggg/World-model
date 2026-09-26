# Easy-Event Membership Diagnosis

## Material Passport

Fresh post-readout diagnostic computation on cached_verified parent predictions
and labels. No new head training or policy evaluation in this diagnostic.
The parent is dcb48b9d, with 58 artifacts and 92 source bindings verified.
The preceding goal turn is progress: 144 fits, a failed primary magnitude gate,
and positive secondary ranking evidence changed the next research question.

## Findings

Split the same positive-disagreement rows into four mutually exclusive groups:
outside-easy/no harm, outside-easy/positive harm, easy/no harm, easy/positive
harm. The easy cut is still fitted on the three training localities only.
Future error is used only to construct diagnostic labels, never a feature.

| Pair | Dependent views | Worse MSE views | Outside-easy dominates worsening | Largest group: outside/no harm | Outside/positive harm | Easy/no harm | Easy/positive harm |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 72 | 39 | 35 | 20 | 15 | 2 | 2 |
| motion-only | 72 | 38 | 34 | 32 | 2 | 0 | 4 |

Outside-easy rows account for 97.097% of positive excess-MSE contributions
summed across the full worsening views, and 99.673% for motion-only. These
sums repeat localities/roles and are descriptive, not independent sample
statistics or a universal probability. Per-view partitions remain available.

This is more specific than saying the new head merely overpredicts harmless
easy examples. Much of the extra easy-harm mass is assigned outside the easy
event. Aggregate harm coverage can improve while this membership/magnitude
allocation worsens. These findings do not prove that an explicit event gate
or any proposed calibration will generalize.

## Next Test

Keep the reference-cost path unchanged. Compare identically trained bounded
harm readouts using the frozen mean versus fractional hidden representations.
This tests whether the representation contains useful transferable information
despite the original multi-output magnitude fit. It is not another threshold
search, global rescale, or direct event-probability claim. Training and primary
readout rules are specified separately in registration.md before fitting.

Primary protocol remains obs8/pred12 annotation steps, image pixels,
detector-derived labels, 2% policy tolerance. No independent-role opening,
new policy, metric/seconds, human-gold, physical safety, true3D or foundation
claim. Stage5C and SMC remain off.
