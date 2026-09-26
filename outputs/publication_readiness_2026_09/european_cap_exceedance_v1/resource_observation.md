# Runtime Placement and Pilot

## Material Passport
Fresh local observation,2026-09-26. No remote training was submitted.
CREATE queue observation succeeded without modifying jobs;private receipt
SHA256837b662e4a3915dcbd54e524ca4fe2c99123f48f2192f5a3aa3b762eb1221f0b.
This is a current queue observation,not evidence about historical job completion.

Native arm64 Python3.11.1/Torch2.12.0;CPU4/interop1/workers0. At preflight,
19GiB disk free,with a10GiB preserved reserve. Support PID17378 completed144
views;observed RSS was approximately12GiB. No model training ran in support.

Pilot PID18169 trained the first registered MLP for100 updates with12,833
parameters. Training time0.088315125s;pilot wall15.978738167s includes first-view
source loading but excludes ancestry preflight. Fixed-batch BCE decreased
0.084219396 to0.076875404;unknown-label rows sampled0. This is not generalization.
The simple100-step projection is508.695120s for576,000 MLP-rate updates;it is
not end-to-end runtime or a guaranteed ETA. Both linear and MLP arms remain
at their registered2000-update budget.

Full training started under PID18461 with explicit resume. The first MLP's
100-step checkpoint is continued,not discarded. Per-head atomic checkpoints,
optimizer/RNG states and heartbeat are retained privately. The private pilot
receipt records the historical100-step hash;that checkpoint path changes on
resume and is not represented as an immutable current100-step artifact.
Completed head receipts freeze the final model and held causal predictions.

The local data position,small heads,pilot cost and memory support local CPU
execution. No need to occupy CREATE GPU capacity for this fixed experiment.
Forecasts,independent data roles,policy thresholds and deployment are unchanged.
