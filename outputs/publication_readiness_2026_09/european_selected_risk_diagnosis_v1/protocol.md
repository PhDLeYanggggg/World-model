# Frozen-Action Residual and Support Diagnosis

## Material Passport
- Role: experiment validation and failure diagnosis.
- Source: sealed European subset-risk experiment; development-only follow-up.
- Parent seal: b855b36149323d05e41d82f6d63dad91636a42f84afc559d5ce85b1f8ef4cb2b.
- Results: post-hoc development diagnostics, not independent confirmation.

## Fixed Questions

1. Do frozen optimizer-selected rows show larger signed-risk underprediction
   than eligible unselected rows and the three causal proxy banks?
2. Do selected rows fall outside the two fitting sources' causal descriptor
   support, or outside the old controller-admission proxy?
3. Does the aggregate objective add harm, remove benefit, or both? Report
   exchanges at matched per-query rank counts separately from joint policies,
   whose counts can differ between arms.
4. Are errors concentrated in particular held development localities, query
   sizes or descriptor-support slices? Source associations are not causal effects.

## Locked Scope

Use all 108 frozen groups, two heads, two held development localities each,
three forecasting seeds and the existing four/four/two/two roles. No new
forecaster, loss, policy, threshold, held selection or deployment occurs.
Prediction APIs receive only the six previously allowed causal data keys.
Targets are read afterward for accounting only. Unknown-label rows remain
in inference, decisions, proxy membership and query sizes, never in error sums.
The original selected-harm 2% screen and its undefined cases are unchanged.

Compute signed excess as positive neural harm minus 0.02 times floor error,
with a second component restricted to the frozen CV-easy definition, both
divided by the fitting-only cost scale. Optimism means truth minus prediction:
positive values mean underprediction. Do not interpret the four head outputs
individually as identified moments, probabilities or uncertainties.

Inspect eligible rows, eligible unselected rows, selected rows, all three proxy
banks, selected rows inside/outside the controller proxy, and selected rows
inside/outside descriptor support. Query sums with incomplete selected labels
are separately counted, not certified. Empty residual sets are undefined.

## Causal Support Descriptor

Use the existing six causal descriptors and fitting-only source-balanced mean
and standard deviation, without clipping for this diagnostic. For each of the
two fitting sources, construct a nearest-neighbor tree from known fitting rows.
Its radius is the 95th percentile of distances from the *other fitting source*.
A held row is outside both sources if both distances exceed their corresponding
radii. This uses no held outcomes or held distribution to fit radii. Also report
outside-either frequency. Overlapping rows are not independent support samples.
This six-dimensional diagnostic is neither full latent support nor calibrated
epistemic uncertainty, and is not a new admission rule.

## Accounting and Uncertainty

For aggregate versus pointwise actions report newly selected, removed and shared
sets, positive harm and benefit separately, normalized by the unchanged total
known floor error. Net gain change must equal added benefit minus added harm
minus removed benefit plus removed harm. Unknowns stay separate. Full-floor
normalization does not replace selected-harm risk or its 2% tolerance.

Reduce dependent views within each of twelve localities first. For prespecified
residual-selection gaps and accounting contrasts, use 3,000 paired locality
bootstrap draws, seed 101531; incomplete locality coverage means no full-roster
CI. These nominal intervals are descriptive, not multiplicity-adjusted claims.
No best-source or best-policy selection from these diagnostics is allowed.

## Execution and Limits

Native arm64 Python, CPU4/interop1/workers0. Checkpoint one aggregate record per
group with hashes; resume validates completed groups. Retain 10GiB free space.
Sealed parent files are never modified. Save only code, protocol, aggregate
reports and lightweight evidence in Git. This protocol is committed before
new diagnostic readout, but known parent outcomes make it post-hoc regardless.

Independent selection, calibration and confirmation remain closed. No metric,
seconds, human-gold, true3D/foundation or physical-safety claim. Main protocol
is obs8/pred12, raw-frame stride12, image-local detector silver. Stage5C and SMC
remain disabled. Diagnostic success does not mean the research goal is achieved.
