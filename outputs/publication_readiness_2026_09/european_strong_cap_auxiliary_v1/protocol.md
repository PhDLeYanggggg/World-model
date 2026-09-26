# Cap-Event Auxiliary on the Reconstructed Strong Cost Model

## Material Passport and Hypothesis
Source-development experiment registered before new support/fitting/readout.
The previous399-input SiLU32 two-cost design failed against the strong original.
It also changed architecture,objective and training support. This experiment
asks whether cap-event auxiliary supervision helps when those original
choices are preserved. It is not a retrospective claim that they caused the
previous failure, nor a decomposition of their individual effects.

## Exact Strong-Base Contract
Use the original383 native causal inputs,original fitting-only preprocessing
without the previous added context/risk features or clipping, GELU64 and four
cost outputs. Keep all known cost rows,including zero-disagreement rows.
Unknown labels remain excluded. Use the original site-balanced samples,
seeds17/29/43,2000 updates,batch256,AdamW0.0003/weight decay0.0001,clip5.
Four-output RMS-normalized MSE,mean/output initialization and the original
unused membership-head intercept are retained. All three arms have identical
parameters,initialization,draws and budgets. That intercept is not a calibrated
cap-event probability; it is retained to reconstruct the original model.

New arms: cost_only,cap_aux,shuffled_aux. Add0 or1 times auxiliary binary log
loss. Labels use common-meta easy harm exceeding a row-locality-excluded
inner teacher's all-harm prediction,only on known positive-disagreement rows.
Masking this auxiliary does not remove zero-disagreement rows from cost loss.
An auxiliary-empty minibatch contributes differentiable zero event loss.
Within-locality shuffling preserves event counts and missingness. Neither
event labels nor teacher scores are inference inputs. Probability never
multiplies the cost outputs. Generic membership auxiliary has already been
studied; this experiment uses the distinct cap-exceedance event.

The first arm in every view must reconstruct the frozen original model state
and composed held predictions within rtol1e-6/atol1e-7. Record maximum absolute
errors and bitwise equality separately; do not call tolerance equality exact.
All original input/target/preprocessing hashes and sampler/RMS states must
match. Stop that view on failure and diagnose implementation, not held efficacy.
This is a fresh original-control retrain,not just cached inference.

## Source Roles and Fixed Matrix
144 views = six producer/controller assignments x three seeds x two forecast
families x four held source localities.432 new heads,864000 updates. Keep all.
Three fitting localities per view. Forecast producers exclude all risk-model
localities. Auxiliary fitting teachers exclude each row's locality. Inner
two-locality versus outer three-locality teacher transport remains unresolved.
Common easy cuts,means,scales and normalization use fitting rows only.
Support check before fitting;100-update runtime pilot then resume to2000.
All predictions are frozen and committed before source-held outcome readout.

No future endpoint,actual future error,event/easy label,scene ID or held
outcome is an input. No endpoint-goal construction. Positive-CV easy excludes
perfect-CV cases; their separate zero-reference guard is unchanged. Original
reference moments D/D_E are frozen in held comparisons,although all four
native moments are learned in the training objective.

## Measures and Gates
Reuse the established primary: positive-disagreement easy-harm MSE gain.
Guards: all-row all-harm MSE gain,top10 easy-harm capture gain and reduction
in absolute log mean predicted/actual harm. Coverage here is a cost ratio,
not conformal coverage. Also report event probability diagnostics,fitting
loss,fit/held differences and original-control reconstruction error.

Average three seeds within locality,then3000 paired resamples of four
localities,bootstrap seed71429. Keep all six overlapping assignment contrasts
and unsupported entries. Intervals are source-exploratory,without multiplicity
correction. Full/motion changes forecasts and outcome populations,not features
alone. The source pool has prior outcome exposure.

Primary gate: all six full-input MSE intervals positive against reconstructed
cost_only and original. Guards: no negative or missing intervals on the three
guard measures against both. Task-information gate: all six full-input primary
intervals positive against shuffled_aux. Cost contribution requires every
gate and successful original reconstruction. No favorable-arm/scene selection,
threshold search or early stopping. A passed cost gate alone is not policy
utility,independent confirmation or permission to deploy.

## Runtime and Boundaries
Native arm64 .venv-pytorch,CPU4/interop1/workers0; no resource probing.
Checkpoint and heartbeat every200 updates; exact resume of optimizer/RNG.
Preserve10GiB free disk. Local-first placement is checked with a real pilot;
CREATE was queried read-only,with no jobs submitted or changed. Slow running
is not a reason to relabel a smaller run as complete.
Independent selection,reserved calibration and confirmation remain unopened.
No new forecasts or policy are evaluated. Obs8/pred12 native annotation steps
and detector-image pixels. No metric,seconds-level,human-gold,physical-safety,
true3D,foundation or submission-ready claim. Stage5C and SMC remain off.
