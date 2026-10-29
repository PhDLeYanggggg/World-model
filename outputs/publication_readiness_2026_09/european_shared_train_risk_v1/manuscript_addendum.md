# Sharing Empirical Risk Learning Does Not Resolve Recording Shift

## Scope and Hypothesis

This addendum extends evidence manuscriptv5 with a complete negative experiment,
not an independent confirmation or submission-readiness claim. We investigate
whether pooling cost learning across source recordings repairs failure of
source-specific risk models. Previously, scaled constraints fit known TRAIN
risks substantially better than their development-recording risks. Most
source/context groups have only two TRAIN recordings.

For each of six frozen forecasting/controller contexts, we partition the TRAIN
recordings of four sources using an outcome-independent hash. All preprocessing,
support statistics, cost scales and priors are refitted on inner-fit recordings.
The held recordings do not determine gradients, checkpoints or thresholds.
The source corpus has prior exposure, and forecast producers are frozen; this
inner partition does not manufacture an independent confirmation population.

## Matched Learning Experiment

We train three32-wide cost-model arms:72 source-local heads,18 shared heads with
global risk multipliers, and18 shared heads with source-specific multipliers.
Three cost-head seeds are used. Each fit receives2,000 updates. Every source
contributes16 sampled queries per update, and its sampling history exactly
matches the local control. Shared arms use identical initialization and differ
only in risk-multiplier grouping. Their comparison to local models additionally
changes parameter sharing, source-normalized decoding priors and loss scales.

The five predicted moments are nonnegative benefit, positive harm, reference
error, easy-reference error and easy harm. Actions require causal support,
positive predicted net benefit and nonpositive predicted excess over the fixed
2% positive-harm/reference budgets. This is a cost-learning study on frozen
forecasts, not a new trajectory generator, JEPA or latent-rollout experiment.

All108 final checkpoints are frozen before the72-view held readout. We retain
unknown outcomes and compare full policies and query-wise count-matched actions.
Pairwise lower utility subtracts the disagreement envelope for unknown outcomes
selected by only one arm; shared unknown selections cancel. The normalized
lower utility is expressed in percentage points of known floor-reference error,
not as an identified ADE/FDE improvement. Different pairwise bounds cannot be
subtracted transitively as point utilities.

## Complete Results

| Contrast | Full lower utility, pp [nominal95% CI] | Same-count lower utility, pp [CI] | Signed-score MSE difference [CI] |
|---|---:|---:|---:|
| Shared global minus local | -2.094903 [-3.479909,-0.824900] | -0.519429 [-1.025350,-0.147155] | 7.845471 [2.612407,15.439524] |
| Shared source minus local | -2.604299 [-4.255594,-1.128204] | -0.381723 [-0.795017,-0.070482] | 3.438174 [0.831990,6.243236] |
| Shared source minus shared global | -1.145644 [-1.672529,-0.637533] | -0.158733 [-0.340291,-0.006414] | -4.407297 [-12.244204,0.217208] |

Known easy-risk violations are46/46/40 of72 for local/global/source, while
complete finite support is0/0/1. Undefined easy ratios are7/8/12. Source-specific
constraints reduce some violations but do not establish safety or useful
transfer. Both repairs fail the registered routing rule and neither is promoted.

The419,766 held row occurrences per arm include9,228 unknown outcomes. A
3,000-draw paired-locality bootstrap averages repeated views within each of12
localities; intervals are nominal and neither simultaneous nor independent
confirmation. Three cost-head seeds are not three forecast-producer seeds.

## Fitting Versus Generalization

A registered post-readout census evaluates all frozen heads on their inner-fit
recordings, without optimizer updates or new held predictions. It includes
897,060 repeated fit row occurrences per arm,18,978 unknown. Known easy-risk
violations on fit/held are1/46 local,10/46 shared-global,9/40 shared-source.
Mean signed-score MSE on fit/held is0.559739/4.013844 local,1.968616/11.859315
shared-global and1.965634/7.452018 shared-source.

The census supports two separate limitations: source-local fitting does not
generalize reliably to other recordings, and the tested shared model already
fits its training costs worse. It does not uniquely identify capacity, source
aliasing, normalization or objective conflict. Reduced held/fit ratios for
shared models are not a performance gain because their denominators worsen.

## Reproducibility and Claim Boundary

Native arm64 Torch used four compute threads and no loader multiprocessing.
The real pilot tested exact interrupted/direct resume. Complete fitting took
489.45s; held readout63.05s. The additional fit census resumed unchanged after
one SSH255 transport interruption; its successful process took77.52s. All108
payloads, matched sampling histories and aggregate summaries verify.41 scoped
tests pass; the full legacy suite was not rerun because some tests modify frozen
research artifacts. Raw inputs and checkpoints are excluded from Git.

```bash
.venv-pytorch/bin/python -m scripts.report_m3w_shared_train_risk --check
.venv-pytorch/bin/python -m scripts.diagnose_m3w_shared_train_fit check
```

Observation8/prediction12 uses raw-frame stride12 and dataset-local coordinates.
Historical t+50 is a different protocol. No seconds, metric, physical-safety,
true3D or foundation-scale claim follows. Silver labels are not human gold.
Independent selection/calibration/confirmation remain unopened by this study;
deployment, Stage5C and SMC remain unchanged. These results are useful negative
evidence, not the missing positive core contribution. The manuscript AI-use
disclosure and final author submission approval still apply.
