# Nested Locality Residual Supervision

## Material Passport

Parent b631bf80 is cached_verified locally and on GitHub:51 artifacts and10
source bindings match. The previous goal turn made progress:864 fixed context
probes were fitted and fully replayed, but only2/6 full MSE intervals improved
against the original and global-bias controls. Their base predictions on
residual-fitting rows were in-sample. This experiment tests a new residual
provenance, not a larger forecaster, threshold sweep or independent calibration.

## Fixed Hypothesis and Controls

Can fitting-locality-OOF error supervision improve the same seven-feature
cost correction on another source-development locality? Keep the original
strong three-locality cost estimator, trajectory producers, eight observed/
twelve predicted annotation steps, outer target rule and deployment fixed.
All144 outer views /six source assignments /three seeds /two input pairs stay.

For each outer view, exclude the outer locality from every fitting component.
The remaining three controller localities define three inner folds. Fit one
cost-only neural head on two localities and predict all three fitting localities.
The group's frozen forecasting producers are disjoint from all controller
localities. Never borrow another group's model that has seen the outer locality.

Rebuild mean/std, equal-locality weights, positive-CV-ADE25th-percentile easy
cut, cost/loss scales, initialization and sampler using only the two inner
fitting localities. Future errors are labels only. Each prediction's residual
uses that producer's own inner-fitting easy cut, never a cut estimated using
the inner-held outcome. The outer evaluation retains its existing three-
locality fitting-only cut. Quantify inner-to-outer cut and label drift.

There are432 new native-Torch cost heads and864000 updates, with unchanged
width64,2000 steps,batch256,AdamW learning rate0.0003 and inherited fixed
settings. These are risk-estimation heads, not new trajectory dynamics or
latent-generative training. No newly selected checkpoint or hyperparameter.

Three residual banks have exactly one producer per row and equal-locality
weighting. For sorted fitting sites a,b,c:

- OOF: rows at j use the model excluding j.
- Matched in-sample-next: rows at j use the model excluding the next site.
- Matched in-sample-prev: rows at j use the model excluding the previous site.

The two cyclic controls both include j in fitting, but match the OOF producer
architecture, two-locality training size, budget and seed. Their assignment is
fixed by roster order, not performance. Both are retained. Different training
compositions and inner cuts remain possible transport effects, so this is not
a perfectly isolated causal estimate of exposure alone. Report that limitation.

For each bank fit the same global-intercept and additive-context ridge0.1
probes as the parent. Keep all seven features, fitting terciles, missing bins,
positive-disagreement fitting support and target-RMS convention.864 closed-form
fits total. Apply corrections only to the frozen original outer predicted H_E,
clipped to[0,original H]. D,H,D_E and both trajectories remain unchanged.
No in-sample feature/cut statistic from an outer-held locality is allowed.

## Frozen Sequence

Commit protocol and registration before support/pilot. Inspect numerical
support on all432 inner fitting sets; no selective fold removal. Require
known labels, both easy strata and positive easy-harm rows. This is numerical
support, not power. Commit support before the first100-step real training
pilot. Resume it within the full2000-step budget. Preserve checkpoints, inputs,
row alignment, sampler state, heartbeat and a10GiB free-space reserve.

Freeze and commit all432 inner predictions before fitting residual probes.
Then freeze and commit all outer probe predictions before outer outcome readout.
Outer source-development outcomes were already exposed historically; these
freezes do not restore independence. Reserved selection/calibration/confirmation
remain unopened. The full local experiment is not replaced by the pilot.

## Readout and Gates

Use the parent's same easy-harm MSE, top10 harm-mass capture and coverage-log-
error definitions on all known and positive-disagreement rows. For every
contrast, first average three seed differences within locality, then resample
four localities3000 times. Retain all six dependent role assignments and both
input pairs; no window-independent CI or multiplicity-adjusted claim.

Primary full-input OOF-context contrast must have six positive MSE intervals
against each of:original; cached three-locality in-sample context; OOF global
bias; matched-next context; matched-prev context. Require no negative/missing
top10 or coverage intervals in these five comparisons. Also retain global and
cyclic controls versus original. The latter guard is not noninferiority proof.
A failed primary cannot be replaced by a favorable secondary comparator.

Even a positive diagnostic does not authorize a deployment or independent-data
advance: calibrated intervention and trajectory utility have not been evaluated.
Reconstruct all inner preprocessing/targets, replay432 checkpoint predictions,
refit864 probes, recompute36 readout groups and independently check held MSE.
Use targeted tests including inner-held perturbation, cyclic routing and exact
two-locality checkpoint resume. Same-version ancestor tests remain cached_verified.

## Resource and Claim Boundaries

Native arm64 `.venv-pytorch`,CPU4/interop1/workers0, no resource probing or
NumPy-as-neural fallback. Local first after pilot measurement; CREATE only if
needed and through the approved scheduler. Existing unrelated jobs untouched.
Original data, checkpoints, per-row predictions and caches stay private.

Detector-derived pixels and annotation steps only. Not verified meters,
seconds, human gold, physical safety, true3D, foundation or submission-ready.
No Stage5C execution, SMC, policy/threshold tuning or new deployment.
