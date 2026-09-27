# Matched Agent-Track Topology Experiment

This is source development, not independent confirmation. Register and commit
this design before fitting. The previous partial-neighbor contrast failed;
its controls remain evidence, not a reason to reselect the evaluation set.

## Hypothesis and Matched Change

Flat past tokens discard supplied cross-time neighbor associations. Keeping
those associations through within-agent temporal attention followed by
between-agent attention may improve prediction. The structural counterexample
does not prove forecast benefit and cannot identify the cause of prior failure.

Reuse the sealed partial geometry, all 318,969 target queries, 163 recordings,
12 source localities, original three four-locality producer folds and seeds
17/29/43. Fit nine models for 4,000 updates each. The first 100-update pilot
resumes into the first endpoint; it is not an additional trial or smaller run.
Reuse batch64, AdamW, native-coordinate loss and locality weighting, exact
sampler sequence, width64/head4/two layers, 88,514 parameters, same initial
weights, training-only baseline choice and bounded causal output.

The ONLY changed mechanism is attention topology: layer one within each
agent's masked past history; masked temporal pooling; layer two between agents.
Time features, conditioning, ego/neighbor support, masks and request queries
remain identical. Pooling and topology are one versioned encoder intervention;
the experiment cannot isolate their individual effects. No smoothing, new
loss, goal input, neighbor cap, risk head, threshold or budget search.

## Controls and Execution

The nine sealed flat partial-neighbor fits are the primary controls. Verify
hashes and recompute their predictions with the current runtime before scoring.
The original complete-neighbor controls are retained as a secondary contrast.
All fit preprocessing excludes each held locality. Verify all new sampling
counts, final sampler states and initial parameters against the primary controls.
Keep original sealed sources untouched; new checkpoints and predictions are
private ignored artifacts. Atomic checkpoint every200 updates, heartbeat every50,
explicit resume. Native arm64 Torch CPU4/interop1, zero workers, no resource
probing. Check CREATE read-only; move only if the real local pilot justifies it.
Preserve 10GiB free disk. Slow is not failed or an excuse to reduce the budget.

Freeze all nine new/flat prediction pairs in Git BEFORE comparative scoring.
Future positions and label-validity masks are loss/evaluation only; inference
uses only the existing pack_geometry input contract. No reserved selection,
calibration or confirmation readout; no held-out outcome-driven seed selection.

## Evaluation

Primary: mean of the 12 locality-specific ADE percentage gains versus matched
flat predictions, each averaging two producer contexts and three seeds. Use
3,000 paired locality bootstrap resamples, seed71431. Report per-seed/fold/site,
ADE/FDE, all/positive-easy/hard/zero-CV, training-selected baseline and CV,
original complete-neighbor neural reference, absolute errors, p95/p99, positive
gain/harm and finite/smoothness proxies. Easy/hard cuts remain training-derived
CV-ADE cuts, including FDE subsets. Never substitute ADE for FDE denominators.
Undefined zero-denominator percentages remain undefined with absolute costs.

Exploratory benefit: primary lower95% interval >0, positive-easy degradation
vs flat <=2%, hard ADE point gain vs flat >=0. Also disclose easy degradation
vs CV even if the primary screen passes. These are forecaster screens, not a
policy safety guarantee or deployment authorization. Source outcomes were
already development-exposed, producer fits overlap and windows are dependent.
Do not pool image-local distances or claim the bootstrap is independent testing.

Predeclared causal slices: no/one/multiple current neighbors; partial-history
present/absent. Report all without choosing the main result from them. They
are diagnostic: a gain without neighbors cannot establish interaction benefit.

## Research Boundary

Task: obs8/pred12 at raw-frame stride12, not historical t+50, seconds, metric,
true3D or foundation evidence. Labels are detector-derived, not human gold.
No deployment change, Stage5C execution or SMC. A supported predictor contrast
would still need matched gain/harm learning, scene-joint intervention and
independent risk calibration. A failure requires evidence-based diagnosis,
not a new name for the same unsuccessful hypothesis.
