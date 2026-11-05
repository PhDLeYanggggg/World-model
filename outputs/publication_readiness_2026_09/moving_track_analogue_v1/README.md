# Cross-Source History Retrieval

I tested whether nearby causal histories from other sources could identify
useful interventions missed by the fitted risk forests. The fixed model
uses one window per source/agent identity bucket and 32 buckets per query.
It excludes the entire query source from references and preprocessing.

**History matching helps relative to a misaligned control, but does not
repair safe intervention.**

| Context | Model | MSE reduction % | Decision-cost reduction % | Admitted | Harmful |
|---|---|---:|---:|---:|---:|
| critic0 | shared64 | +0.000000 | +0.000000 | 1 | 0 |
| critic0 | prior | +8.034090 | -0.033689 | 0 | 0 |
| critic0 | track32 | -0.426663 | -4.481672 | 979 | 107 |
| critic0 | misaligned32 | -2.916355 | -17.995976 | 979 | 132 |
| critic1 | shared64 | +0.000000 | +0.000000 | 10 | 1 |
| critic1 | prior | +2.868915 | -0.020976 | 0 | 0 |
| critic1 | track32 | -0.198968 | -121.691096 | 1762 | 723 |
| critic1 | misaligned32 | -0.483684 | -180.438381 | 1762 | 772 |
| critic2 | shared64 | +0.000000 | +0.000000 | 0 | 0 |
| critic2 | prior | +30.093705 | +0.000000 | 0 | 0 |
| critic2 | track32 | +33.419794 | -2.568241 | 2165 | 521 |
| critic2 | misaligned32 | +24.394702 | -21.490365 | 2165 | 794 |

Positive reductions are better; reference is the shared64 forest.
The registered information screen passes **0/3** contexts.
Two were required. Source-macro decision costs are not trajectory ADE
improvements or the full scene-level safety estimator. No model is promoted.

## Matching Versus The Null

| Context | Cost reduction versus misaligned matching % |
|---|---:|
| critic0 | +11.453190 |
| critic1 | +20.948376 |
| critic2 | +15.575000 |

The fixed within-source permutation preserves the unweighted prediction
multiset and admission counts, not all query-weighted moments. It is one
null draw, not a significance test. Better than this null does not imply
better than the strong control or compliance with the original risk rules.

## Every Excluded Source

| Source | Known rows | Track32 admitted | Harmful | Selected normalized easy harm/reference % |
|---|---:|---:|---:|---:|
| moving_source_component_048 | 1769 | 891 | 106 | +9.715444 |
| moving_source_component_067 | 3998 | 88 | 1 | +8.045716 |
| moving_source_component_053 | 672 | 27 | 1 | NA |
| moving_source_component_065 | 44 | 0 | 0 | NA |
| moving_source_component_076 | 2999 | 1735 | 722 | +72.430162 |
| moving_source_component_059 | 8772 | 376 | 100 | +14.620657 |
| moving_source_component_060 | 10749 | 580 | 189 | +25.300310 |
| moving_source_component_073 | 19058 | 1209 | 232 | +14.598378 |

These ratios use only known selected proposal labels, with causal row
normalization and source/query weighting. They are descriptive, not the
full recording-level physical risk or a replacement for the support gate.
NA stays NA, including admitted rows without easy reference support.

## Evidence And Scope

- fresh_run: lineage metadata and all 48,061 source-excluded risk predictions.
- cached_verified: original 120-feature caches, targets, weights and eight shared64 controls.
- not_run: new neural trajectories, full-policy evaluation, new nested-C/H and independent confirmation.
- All eight complete retrievals reproduce exactly with a different chunk size.
- The 1,274 unknown proposals remain audited, not assigned safe labels.
- Identity buckets may merge different people; relabeled duplicates may remain.
- Overlapping windows and identity buckets are not independent samples.
- Original 2% positive-harm/easy-net and support requirements are unchanged.
- No identity-fallback certification. Original24-group primary remains NA, not passed.
- Reserved Moving20, DUT and DroneCrowd remain closed.
- Native obs8/pred12 image coordinates and silver labels; no metric or seconds claim.
- No true3D, foundation, Stage5C, SMC, deployment or submission-ready claim.

[Protocol](protocol.md) | [Distance diagnostics](distance_analysis.md) |
[Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)
