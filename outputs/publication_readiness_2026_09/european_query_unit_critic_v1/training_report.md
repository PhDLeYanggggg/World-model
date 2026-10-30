# One-factor query-unit repair: real training

fresh_run:36 complete real Torch critics,72,000 updates plus4,000 separate exact-refit updates.
cached_verified:36 original critics,unchanged causal feature/source inputs,forecasts and sampling design.
Training PID55207: 299.778s,peak RSS10450976768B.
Native arm64 CPU4/interop1/workers0. Both full first-head refits agree exactly.
Only the query-score denominator changes. All original random draws,initial states,architecture,
pointwise loss,0.5/0.5 weights and allocation rules stay fixed. Unknown labels never supervise.

## Common-unit fitting telemetry

Each curve monitors16 fixed source-fitting queries,not validation/test or full-dataset means.
New query scores below are converted to ORIGINAL units before comparison. New total loss is
not numerically comparable to old total loss. Step2,000 is fixed;these curves select no checkpoint.

|Arm|Loss units during training|Initial original-unit query score|Final original-unit query score|Heads decreased|Final point loss|Query/point gradient median|
|---|---|---:|---:|---:|---:|---:|
|row_local|old|0.000435536|0.000443910|8/18|0.071484236|0.0008093232204893755|
|row_local|new|0.000435536|0.000144110|15/18|0.078885021|1.7576483894424197|
|selected_set|old|0.000435536|0.000448284|10/18|0.064269093|0.0007081863900496629|
|selected_set|new|0.000435536|0.000153624|16/18|0.078215335|1.6346242439618357|

Gradient ratios compare unweighted terms on the same actual update batch;both weights are0.5.
Increasing the query gradient is a manipulation check,not a safety certificate or unique causal explanation.
Original selected-set fitting may still differ from the final reoptimized policy selection.
Protected-output finite-label interpolation limits are reported separately and not relaxed.

## Boundaries

Exposed SOURCE fitting/resubstitution only,not whole-pipeline OOF or independent confirmation.
Task:obs8/pred12 at raw stride12;detector-silver/local-image coordinates.
No new trajectory forecaster,metric,seconds-level,human-gold,true3D or foundation claim.
No deployment change,Stage5C execution,SMC or submission-ready claim.
