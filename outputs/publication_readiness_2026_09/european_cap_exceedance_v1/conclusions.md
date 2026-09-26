# Cap-Event Learnability: Conclusions

## Question and Result

There is useful source-development evidence that causal inputs predict whether
positive-CV-easy excess forecasting cost exceeds a frozen all-harm estimate.
This is a different, narrower result than learning the magnitude of harm,
improving trajectories, or deploying a safe intervention policy.

The complete registered MLP signal gate is **false**. Two required intervals
overlap zero. The positive linear-control results must not be hidden, and the
failed complete gate must not be changed after readout.

## What Was Run

- fresh_run: 288 native-Torch heads, 576,000 effective fixed updates, 144
  source-held locality readouts, descriptive support analysis and figures.
- cached_verified: frozen trajectory predictors, inner OOF risk teachers,
  outer risk estimators, source rows and their recorded lineage.
- not_run: new trajectory training, policy evaluation, independent selection,
  reserved calibration, confirmation, and the full legacy test suite.

Both arms use the same 399 causal inputs, preprocessing, examples and budget.
They differ in linear logistic versus SiLU32 nonlinear parameterization.
The first 100-update pilot checkpoint was resumed to 2,000 updates, not counted
as another complete training run. All 288 models completed normally. The main
training phase took 635.35 seconds including source loading and inference,
excluding ancestry preflight and subsequent verification. This is measured
runtime, not a claimed end-to-end paper experiment cost.

Registration 21299404 preceded support; implementation amendment 7ae3939d
preceded fitting. Prediction freeze 882a170f preceded held-label readout.
The amendment enforces the existing one-class ranking rule, not a new
scientific criterion. No held outcome selected a checkpoint or threshold.

## Positive Evidence

For full-input linear probes, all six assignment intervals favor the probe
over the fitting-only constant prior in binary log loss and Brier score.
Log-loss point gains range from 0.03458 to 0.12157 nats. All six AP intervals
favor the probe over causal disagreement ranking: point gains are 22.77 to
32.65 percentage points. All six also favor it over each frozen risk-fraction
ranking control. This is more than merely fitting the training labels.

Full-input MLP probes have six favorable Brier and AP intervals, but only five
favorable log-loss intervals. Five top-10%-overshoot-capture intervals favor
the MLP over disagreement; the sixth overlaps zero. Linear capture has four
favorable intervals and two overlaps.

The descriptive median held AUROC is 0.8301 for full-input linear and 0.8442
for MLP. These are summaries of 72 dependent views per arm, not independent
replications or one pooled population estimate.

## What Did Not Pass

Two full-input MLP primary components fail the all-positive-interval rule:

| Producer / controller | Contrast | Point | Locality bootstrap interval |
|---|---|---:|---:|
| 0 / 1 | Log-loss gain over prior, nats | 0.01416 | [-0.00756, 0.03226] |
| 0 / 2 | Overshoot capture gain over envelope, percentage points | 15.32 | [-12.23, 41.38] |

Against linear, all six full-input MLP log-loss intervals overlap zero; AP
has two favorable intervals, one unfavorable interval, and three overlaps.
The experiment does not show stable nonlinear superiority.

Motion-only probability estimation is weaker. Four of six MLP log-loss
intervals favor linear; four MLP Brier intervals favor the constant prior.
Three motion-only held views have no positive events, making their ranking
metrics not_estimable. They remain in the evidence, not converted to zeros.

## Statistical and Scientific Boundary

Each contrast averages seeds 17/29/43 within locality, then uses 3,000 paired
resamples of four localities. The six assignments overlap. Intervals are
exploratory source-development intervals without multiplicity correction.
Point ranges above are six point estimates, not one confidence interval.
Repeated windows and dependent views are not independent samples.

Full and motion-only families change the forecast pair and event population;
their difference is not a matched scene, interaction, or JEPA ablation.
Two-locality fitting teachers and three-locality held producers remain a
transport mismatch. Marginal probability agreement is not conditional
calibration. Event probability does not measure expected tail severity.
The positive-CV-easy event does not cover the separately protected zero-CV
stratum. Causal means past-only information here, not causal identification.

No new deployment is warranted. Independent selection, reserved calibration
and confirmation stay unopened. Obs8/pred12 are native annotation steps in
detector-image pixels. No metric, seconds, physical safety, human-gold,
true-3D or foundation-model claim follows. Stage5C and SMC remain off.

## Consequence for the Next Experiment

Test a matched joint-cost model with versus without cap-event auxiliary
supervision, preserving the same forecasts, causal inputs, seeds, budget,
nested cost constraints and zero-reference guard. This is a proposed next
registered experiment, not a result of this one. Do not stack in-sample event
predictions into a new model without a deeper teacher-lineage exclusion audit.
The auxiliary can instead be a training label for a shared encoder, never an
inference feature. Improvement must subsequently survive cost-magnitude,
tail-coverage and equal-intervention scene-joint utility tests. The current
study gives a reason to test that repair, not a reason to deploy it.

See [all results](results.md), [failure analysis](failure_analysis.md),
[figure](cap_event_contrasts.svg), [protocol](protocol.md) and
[project gap](project_gap.md). Verification is recorded separately in
`verification.json`; absence of that receipt means replay is not complete.
