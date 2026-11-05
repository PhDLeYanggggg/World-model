# Lower Regression Error Does Not Establish Better Intervention

## What Was Run

Three real neural risk-head continuations optimize the complete own-B empirical
query-weighted objective with LBFGS. All 600 outer calls change parameters.
The first complete 200-call fit repeats exactly; all three 3+3 versus 6-call
resume tests and initial/final fitting-metric replays match exactly. The three
fitting populations contain 6,197 / 5,521 / 50,748 rows. These are overlapping
forecast windows, not independent people or recording units.

Training finishes in 617.61 seconds, peak RSS 6.62 GB, on native arm64 Torch
CPU4/inter-op1/workers0. The experiment is not interrupted, downgraded or
reported from a NumPy fallback. Each fit reaches the registered 200-call budget,
not the gradient-stationarity criterion. Remaining optimization error is not
ruled out. Solver, precision and decay handling change together; this is not
an isolated one-factor optimizer comparison.

## What Improved And What Failed

| Fitting context | Mean MSE reduction | All decision-cost change | Easy decision-cost change | Result |
|---|---:|---:|---:|---|
| critic0 | 31.98% | +8.32% | -4.21% | all cost worsens |
| critic1 | 37.90% | -1.48% | +2.37% | easy cost worsens |
| critic2 | 41.13% | +2.57% | +6.34% | both costs worsen |

Positive decision-cost change means worse. These are fitting risk quantities,
NOT trajectory ADE, FDE, test easy degradation or a new t50 result. The losses
predict normalized signed budget excess; sign cost weights a wrong decision
by the magnitude of its observed target. Neither is a calibrated safety proof.

The prospective fitting screen fails in all three contexts. Therefore no new
V or H predictions are evaluated. The evaluate command records `not_run`;
its exit0 is not an evaluation pass. No threshold is adjusted to rescue a
result. Earlier held failures remain failures and deployment is unchanged.

## Which Mistakes Remain

Fresh, exactly replayed own-B attribution separates false-safe decisions from
false rejections. It uses only fitting labels and does not choose a model.
The observed-label masks also exactly match the frozen original checkpoints.

- critic0: all-channel false-safe cost falls 35.03%, but false-rejection cost
  rises 17.22%; rejecting useful interventions outweighs the safety-side gain.
  Source048 supplies the positive all-cost change, while067 improves. Fewer
  false-safe outputs (127 to99) do not establish lower total decision cost.
- critic1: all-channel false-safe cost rises 47.50%, even as false-rejection
  cost falls 17.98%. Source076 dominates the added false-safe cost. Easy
  decision cost rises in each of its three fitting sources; its aggregate
  all-cost reduction does not satisfy easy preservation.
- critic2: false-safe costs rise 4.37% all and7.21% easy. Sources060 and073
  account for the increased easy cost;059 improves easy cost. Easy false-safe
  output count rises from952 to1,062. This is not a count of distinct people.

The decomposition also identifies how the regression objective improves:
40.19%,49.92%,75.14% of the total MSE reduction respectively comes from the
unknown-outcome conservative-bound component. Observed-target MSE also falls
in every context; it would be wrong to say only the bounds improve. The
bound-loss contribution is not evidence that bounds are invalid, should be
removed, or causally explain the wrong signs. Bounds remain in training and
all original evaluation/support requirements remain intact.

For each source/channel, the increase in sign cost equals newly introduced
wrong-decision cost minus repaired wrong-decision cost. This identity and the
known-plus-unknown MSE decomposition are checked numerically. Full source
contributions are retained in `fitting_error_attribution.json`; no source is
excluded after looking at its outcome.

## Diagnosis And Next Model Change

Supported: improved finite-sample risk regression does not consistently improve
the decision boundary, even on fitting sources. The optimizer is genuinely
updating and reducing its objective, but this optimization change alone fails
the prerequisite for another exposed-development readout.

Not established: global convergence, irreducible feature ambiguity, causal
responsibility of missing-label bounds, held-source failure of these new
weights, or a successful replacement loss. No new held test was run. The
earlier signed-cost-only experiment already improved fitting signs but failed
held safety; simply repeating that objective is not a justified repair.

The next narrow model hypothesis is to separate risk-magnitude estimation from
the intervention-decision representation while retaining conservative-bound
supervision. Compare it against the same fitted risk control with fixed source
roles and capacity accounting. First require useful fitting decisions in both
channels; only then use the existing nested development readout under the
unchanged original rules. This proposed model has not been trained here. A
larger model, new modality or additional training alone is not claimed to fix it.

## Evidence Boundaries

Fresh: neural continuation, real resume/refit, fitting-metric replay and error
attribution. Cached_verified: original trajectories, proposal policy, source
roles, preprocessing and starting checkpoints. Not_run: new held prediction,
independent confirmation, multi-seed transfer and a new trajectory model.

The original 24-group primary remains NA because reference support is missing.
That data limitation is not repaired by better fitting. Missing support and
empty controls are not passes. No identity-fallback certificate, relaxed 2%
limit, removed support rule, reserved-data use or deployment upgrade.
Native obs8/pred12 image pixels, not metric, seconds, true3D, foundation or
human-gold claims. Stage5C and SMC remain disabled. Not submission-ready.
