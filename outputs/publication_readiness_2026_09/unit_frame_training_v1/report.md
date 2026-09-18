# Observed-Unit Conditioning: Completed Research Check

## Result

The fixed comparison completed **27 fresh neural fits and 162,000 optimizer
updates**, plus nine hash-verified previous controls. None passes both positive
forecast gain and easy preservation. No model is promoted. This is a completed
mechanism experiment, not a completed world-model or submission goal.

Main task: offline supplied annotated histories, eight observed and twelve
predicted native annotation steps. Dataset-local coordinates, no verified metric
or common effective time. Main cohort: 11,966 fit windows; ETH, Hotel and grouped
Zara are three physical-site folds. Primary remains past-normalized ADE with
equal-site aggregation. All folds were already exposed during research.

Source: 229,333 past-supported pedestrian query windows from original SDD
train-40, stride12, twelve targets through +144 raw frames. Source timing is not
physically matched to the main task. Neighbor context may contain other agent
types; no claim of time/scale equivalence follows. Source val/test and all main
development/calibration/confirmation roles remain closed.

## Fixed Comparison

Each fit uses seeds 17/29/43, the unchanged small geometry branch, 2,000 source
updates then 4,000 main updates, batch 64, CPU 4/inter-op 1/workers 0. Optimizer and
main sampler reset at the phase boundary. A 100-update real pilot, without held
evaluation, resumes into the full matrix. Full invocation adds 161,900 updates;
the pilot adds 100. Training is not a full epoch of every source row.

| Arm | Change from old geometry control | Gain vs CV | Descriptive site interval | Safe positive fits |
| --- | --- | ---: | --- | ---: |
| Legacy | Verified prior source/main model | -0.80523% | [-7.22531%, -0.22560%] | 0/9 |
| Unit inputs | Reconstructed dimensionless inputs and canonical direction; original-scale delta | -0.68730% | [-9.45159%, +0.23118%] | 0/9 |
| Unit decoder | Same inputs, radius-decoded delta, original log-ADE loss | -2.48928% | [-5.51996%, -0.07793%] | 0/9 |
| Internal loss | Same unit decoder, log(ADE/radius) in both training phases | -185.77715% | [-408.63852%, -18.87011%] | 0/9 |

CV is also the train-selected strongest candidate in all three folds. These
numbers are not raw-frame t+50 or historical Stage37 results. The 2,000-draw
bootstrap resamples only three exposed physical sites; it is descriptive and
cannot provide independent confirmation. All seeds and site failures are kept.

## What Changed and What Did Not

The internal frame uses only past ego/neighbor geometry and past times, and
reconstructs speed, acceleration, curvature, interaction and baseline-rollout
features. v2 fixes a native-speed cutoff inherited through old turn rollouts.
Past-normalized evaluation targets and evaluation baselines are unchanged.
Future labels enter loss and evaluation only, never the frame or prediction.

No-anchor rows retain exact CV fallback: eight main, 68 source. Source has 225,718
rows with at least one future label; 225,652 also have an identifiable radius
for internal loss. No-anchor rows remain in sampling and evaluation. The old
model with only this guard changes aggregate gain from -0.80523% to -0.80027%:
support guarding alone does not explain the conditioning contrast.

## Failure and Local Signal

Unit inputs reduce static-stay harm and improve the stopping slice by 9.58344%
against CV. Hotel is positive for all three seeds (+0.19954% to +0.25208%), but
easy degradation still fails and Zara worsens to a seed-average -9.45159%.
The overall +0.11699% difference against legacy has a site interval crossing
zero. This is not stable cross-site improvement.

The radius decoder with original loss clips every logged source gradient and
99.44% of logged main gradients. The internal-loss arm avoids this clipping but
greatly worsens primary error and easy harm. Optimizing a smoother objective
did not optimize the protected evaluation task. Absolute easy harm ranges are:

| Arm | Easy absolute normalized ADE increase across nine fits |
| --- | --- |
| Legacy | 0.03505 to 0.18985 |
| Unit inputs | 0.01799 to 0.23946 |
| Unit decoder | 0.05915 to 1.44244 |
| Internal loss | 8.06155 to 96.84135 |

Very large relative easy percentages reflect near-zero CV error, but the
absolute increases confirm genuine damage rather than a percentage-only issue.
Loss magnitudes across primary/internal coordinates are not comparable.

## Oracle Is Not a Policy

A separately labeled post hoc diagnostic lets future labels choose among CV,
legacy and the three new predictions for each row, within each seed. Its ceiling
is +2.63722% versus CV, compared with +0.33196% for CV plus legacy only. These
are labeled-fit oracle values, not learned gains, deployment results, or proof
that a causal gate can recover them. Even this expanded whole-trajectory choice
set has limited aggregate headroom. No oracle labels enter inference.

## Verification and Next Step

27/27 exact checkpoint prediction replays; 82 immutable artifacts unchanged on
completed resume; zero additional optimizer updates. Sample counts and final
sampler states match paired legacy controls. Weights and logged losses/gradients
are finite. 24 focused tests pass; the full unrelated legacy suite was not rerun.
Summed new fit time 277.02 seconds; full-invocation wall time about 290.1 seconds,
excluding pilot and subsequent replay/analysis. Local CPU is appropriate for
this scoped geometry study; no new CREATE job or remote status claim.

Next work should separate feature conditioning from the unsafe decoder/loss
changes and assess identifiable benefit/harm on training roles. Do not expand
this failed internal-loss recipe into a large multimodal run, select thresholds
on these held folds, or pretend that the oracle is a recovered contribution.
Retain the main metric and sealed roles; independent confirmation is still
missing. Any new multimodal comparison needs its own frozen matched controls.

Stage5C and SMC remain off. No true-3D, foundation, metric, seconds-level,
calibrated safety, deployment or submission-ready claim is supported.

See [contrasts](contrasts.md), [failure analysis](failure_analysis.md),
[reproduction](reproducibility.md), [full metrics](report.json),
[loss traces](loss_trace.csv) and [oracle diagnostic](oracle_diagnostic.json).
