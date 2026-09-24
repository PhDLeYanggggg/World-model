# Matched Source-Only Dimensionless Risk-Head Experiment

## Question and Fixed Scope

Does removing explicit native-unit risk features preserve useful intervention
learning on the already design-exposed SDD source population? This is the next
training experiment after the IMPTC input-unit diagnosis, not an admission of
IMPTC or a new external benchmark. Independent confirmation remains closed.

Important terminology correction to the preceding input-contract report:
its first two frozen outputs were easy-weighted harm and easy-weighted benefit,
not overall benefit/harm. Its `net_gain_sign_changed` fields count changes in
the sign of **signed easy risk** (harm minus benefit). The numerical counts
are unchanged. No complete policy or observed future gain was evaluated.

Fix all existing SDD geometry, past normalization, predictor checkpoints,
nested source-excluded forecasts, targets, support, source splits and row draws.
Do not integrate the new prefix canonicalizer in this experiment: that would
change forecast semantics and confound a head-feature comparison. Its unresolved
EqMotion rounding failures remain separate. The primary task stays obs8/pred12
native annotation steps, stride12, annotation pixels, not raw t50 or seconds.

## Training Matrix

Three actions (damping .05, Transformer, EqMotion), four outer source sites,
three seeds and two feature arms give **72 fits**, each 128 ExtraTrees trees,
depth16, minleaf64, feature fraction1/3 (118 columns per split in both arms),
CPU4, single-process loading, checkpoint every16 trees. Preserve the frozen
768,000 source draws per view/action and exclude unknown-label samples.
Runtime pilot: first coupa/seed17/damping/native fit at16 trees, then exact resume
to128 and finish the entire matrix. No budget/model/threshold selection on errors.

Both arms predict the same six bounded fractional targets in one multi-output
forest: total benefit/D, total harm/D, easy harm/D, easy benefit/D,
easy CV/cutoff, and easy probability. D is mean causal forecast disagreement;
zero-D rows cannot intervene. The easy cutoff remains the frozen training-only
native source cutoff. Unknown-label target storage zeros carry zero fit weight.
Total costs come from nested source-only predictions, not in-sample teachers.

`native` has the existing356 features. `dimensionless` keeps the first354,
replaces `log1p(native D)` by `log1p(D/past_scale)`, and removes log native scale,
yielding355 features. Source-only means/stds are refitted for each input arm.
All loss targets, draws, seed and fit settings are matched. Six-output native
control is trained fresh; old four-output heads are not passed off as matched.

## Decision and Readout

Freeze all fits, scores and choices before the new held-source aggregate readout.
The legacy source loader may reconstruct previously exposed arrays; none of the
outer rows may contribute to fitting, feature statistics or threshold choice.
Use fixed predicted net gain (total benefit minus total harm)*D, moving-history
support and positive predicted easy denominator. Rho remains .02.

For each new arm compare: per-agent signed-easy-risk gate; same-query signed
population allocation; same-query clipped-risk selected-denominator allocation.
Also run dimensionless population allocation with the native population's exact
query intervention count, fail closed if infeasible, and report every failure.
Include unchanged CV floor, uncontrolled forecasts and frozen old strict choices.
No new interaction term is introduced. Numeric feasibility/optimality limitations
must remain visible; an estimated risk constraint is not a safety certificate.

Report all windows with available-label ADE/FDE, complete/hard/positive-easy and
zero-CV subsets, unknown/incomplete selected counts, intervention rates, fitting
loss, held-source six-moment error and per-scene/per-seed results. Use the fixed
3,000 paired physical-site bootstrap with four design-exposed sites; intervals
are conditional development evidence, not independent confirmation. Report
partial-future full-grid bounds from the verified parent without imputing labels.

Primary contrast is dimensionless vs native population allocation. Also report
matched-count, pointwise and selected-denominator contrasts, and each new arm
against old strict. Retain all negatives; no readout-based deployment winner.
The experiment can falsify utility preservation but cannot establish external
generalization, calibrated safety, EqMotion numerical repair or publication
readiness. Native source easy cutoffs and raw prefix tolerances remain obstacles
to a fully unit-invariant external deployment contract.

## Integrity and Execution

Only local .venv-pytorch arm64 is permitted. Check hashes, enforce one writer,
emit PID/heartbeat, save resumable forest checkpoints and complete receipts.
Retain all expensive fits on interruption. Run a deterministic resume regression
and score/decision/aggregate replay. Source data, feature matrices, forecasts and
checkpoints remain private and ignored by Git. No external/DroneCrowd outcome,
Stage5C, SMC, metric/seconds claim or change to deployment is authorized here.
