# Causal Output-Bound Diagnosis

Post-fit geometry-only relaxation on previously opened source development.
No new training, model selection, inference features or deployment change.

The bound and causal CV are computed from observed history only. Future labels are used 
solely to compute the independent-step closed-ball ADE infimum. This is a label-aware 
oracle, not a neural prediction; finite-network capacity and causal learnability are not proved.

| Trial | Rows | CV strictly inside or exact fixed | CV outside | Boundary/ambiguous | Zero-motion budget |
|---|---:|---:|---:|---:|---:|
| single0_seed17 | 202146 | 202146 | 0 | 0 | 0 |
| single0_seed29 | 202146 | 202146 | 0 | 0 | 0 |
| single0_seed43 | 202146 | 202146 | 0 | 0 | 0 |
| single1_seed17 | 295207 | 295207 | 0 | 0 | 0 |
| single1_seed29 | 295207 | 295207 | 0 | 0 | 0 |
| single1_seed43 | 295207 | 295207 | 0 | 0 | 0 |
| single2_seed17 | 140585 | 96842 | 43743 | 0 | 0 |
| single2_seed29 | 140585 | 96842 | 43743 | 0 | 0 |
| single2_seed43 | 140585 | 96842 | 43743 | 0 | 0 |

## Formerly Unsupported TRAIN Views

| Context / record | Rows | CV representable | CV outside | ADE lower-bound gain vs CV (%) | New candidate gap to geometric infimum (pixels) |
|---|---:|---:|---:|---:|---:|
| single1_seed43_controller2_dimensionless / eu-locality-082/51 | 78 | 78 | 0 | 93.98831540016637 | 23.39758398465049 |
| single2_seed43_controller1_dimensionless / eu-locality-082/51 | 78 | 78 | 0 | 94.54012594266008 | 28.26326484268129 |

Representability of CV does not establish a safe neural policy. Remaining failure 
may involve learning, context ambiguity or finite model capacity. The geometric oracle is 
neither a model result nor evidence of generalization. Image pixels and raw frames only.
