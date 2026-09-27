# Matched Dimensionless Correction-Fraction Experiment

Source development only. Commit this design before training. The grouped
topology experiment found a small matched gain but unacceptable easy harm vs
CV. Its frozen outputs are controls, not independently confirmed deployment.

## Hypothesis

The old bounded wrapper applies coordinate scale inside a nonlinear correction
fraction: B(x) + R(x) * squash(S(x) * f(x/S(x))). R already supplies coordinate
units. Change ONLY that wrapper to B(x) + R(x) * squash(f(x/S(x))). This removes
the measured unit sensitivity when the inherited input scale clamp is inactive.
It does not prove forecast improvement; it also changes optimization and effective
correction amplitudes. Do not attribute an outcome solely to mathematical invariance.
The one-unit conditioning clamp remains unchanged, so arbitrary very small units
are not covered by the equivariance claim.

## Fixed Data and Training

Use the sealed partial geometry: 318,969 target queries, 163 recordings,
12 source-training localities, three four-locality producer folds, seeds17/29/43.
Nine fresh models, 4,000 updates each, same width64/heads4/two layers and88,514
parameters. Match all initial tensors, batch64, optimizer, schedule, native-coordinate
loss, training-only baseline choice, easy/hard cuts, sampler sequence and final
draw counts. The 100-update pilot resumes into the first endpoint. No extra
trial, seed selection, threshold search, new risk head or loss change.

Grouped temporal/interaction topology, causal budget, histories, masks, queries
and future labels stay fixed. Fit-locality preprocessing only. Future positions
and valid masks enter loss/evaluation only, never inference. Selection, independent
calibration and confirmation roles remain closed. Do not touch historical teacher
oracles, test endpoints, central velocities or test normalization.

## Execution and Freeze

Hash-check all previous sealed code and artifacts; verify each checkpoint and
prediction. Recompute grouped-control inference and require exact cache agreement.
Native arm64 Torch CPU4/interop1, workers0; no resource probing. Check CREATE
read-only; use the real local100-update pilot to assess speed and peak memory.
Checkpoint every200updates, heartbeat every50, explicit resume; keep10GiB free.
Do not shorten the budget because a run is slow. Preserve existing jobs.

Freeze all nine candidate/control prediction pairs in Git before comparative
scoring. Keep checkpoints, geometry, row-level forecasts and runtime receipts
private and ignored. Controls are cached_verified with fresh inference, not fresh
retraining. Register and freeze errors fail closed.

## Evaluation

Primary: equal-locality ADE percentage gain versus matched grouped neural.
Average the two producer contexts and three seeds inside each locality, then
3,000 paired locality bootstrap draws, fixed seed71431. Report every seed and
locality;12 localities are clusters, not1.9million independent samples. Source
outcomes were development-exposed, so this interval is exploratory.

Report ADE/FDE all/positive-easy/hard/zero-CV; training-selected causal baseline,
CV and prior flat neural secondary comparisons; absolute errors, p95/p99,
positive harm/gain, finite and raw-step smoothness proxies. Keep training-derived
CV-ADE subset cuts even for FDE; use correct endpoint denominators. Undefined
zero-denominator percentages remain undefined with absolute costs.

Predeclared slices: no/one/multiple neighbors, partial history present/absent.
No slice selection to replace the primary. Check coordinate rescaling at
0.25/0.5/2/4 using the same observed-history-only prefixes and unclamped support
as the previous diagnostic; report failures rather than silently widening tolerances.

Exploratory predictor screen: primary lower95%CI>0; easy degradation vs grouped
<=2%; hard point gain vs grouped>=0. Separately disclose easy harm vs CV. No
passing predictor screen authorizes deployment or proves calibrated safety.
Require exact prediction/scoring replay and scoped tests before final reporting.

## Remaining Method Contribution

This is a conditioning repair, not a novel world-model method. The main research
question still requires matched relative gain/harm learning, agent-independent
versus scene-joint intervention, equal risk/rate comparisons and independent
scene calibration. Do not keep treating predictor repair as the whole contribution.
No new scene-image/goal/JEPA lift or neighbor-specific ablation is established here.

Obs8/pred12 at raw-frame stride12; detector-derived silver trajectories,
image-local coordinates, not seconds/meters, human gold, true3D, foundation,
historical t+50 or physical safety. No deployment change, Stage5C or SMC.
