# Constraint Feasibility on TRAIN Does Not Establish Selective Forecast Safety

## Scope

This addendum extends the decision-risk experiment and evidence manuscriptv5.
It reports a completed, registered source-development intervention, not an
independent confirmation. The paper remains not submission-ready. Forecast
producers are fixed; only neural cost heads are newly trained.

## Hypothesis and Design

An earlier frozen-TRAIN gradient audit showed that the median weighted easy-risk
gradient was0.0698% of the regression-plus-regret gradient. We therefore test
whether poor constraint conditioning explains the earlier failure. For each
head, positive budget scales are estimated from TRAIN reference costs only.
Both the primal empirical-risk penalty and dual update divide excess by these
constants. The sign of the TRAIN constraint remains unchanged.

The experiment has72 matched head identities for each of two arms: unit control
and budget-scaled risk. All144 fits are fresh, with2,000 updates each. Each unit
control exactly reproduces the preceding task-risk model and optimizer. Inputs,
sampling, initialization, loss components, temperature schedule, hard rule and
2% risk budget stay fixed. Quadratic and utility-only neural controls comprise
144 cached-verified heads; four forest controls are retained. Three cost-head
seeds are not three independent forecast-producer seeds.

The2% budget is positive selected easy harm divided by selected easy reference
error, not net easy ADE degradation. Unsupported/empty cohorts cannot pass.
Unknown outcomes are retained in conservative completion bounds. The registered
rule requires all eight primary full/same-count lower-utility contrasts against
unit, quadratic, utility-only and original forest controls to have positive
nominal lower confidence endpoints, together with complete risk/support checks.

## Data and Uncertainty

Observation8/prediction12 uses a12-raw-frame stride. Coordinates are dataset-local;
neither seconds nor physical scale is established. Historical raw-frame t+50 is a
different task. The72 views cover12 localities and596,988 repeated row occurrences,
including14,076 unknown-outcome occurrences. These are not independent samples.
We average repeated views within locality and use3,000 paired-locality bootstrap
draws. Intervals are nominal, not simultaneous or independent confirmation.

## Results

Values below are scaled-minus-control normalized paired lower-utility percentage
points. They are not ADE/FDE changes. Conservative pairwise bounds depend on the
two policies and cannot be subtracted transitively as point utilities.

| Control | Full policy [nominal95% CI] | Same intervention count [CI] |
|---|---:|---:|
| Unit control | -1.890975 [-3.157684,-0.782172] | -0.484529 [-0.827914,-0.142285] |
| Quadratic | -0.184760 [-1.170105,0.621735] | 0.185634 [-0.134510,0.524161] |
| Utility only | -2.940035 [-4.350033,-1.682244] | -0.699983 [-1.134574,-0.257939] |
| Original forest | 2.258131 [0.718677,3.848069] | 0.778894 [0.252400,1.448081] |

The positive forest comparison does not rescue the stronger-control failures.
The complete signed-score MSE contrast versus unit control is+2.370133
[0.797264,4.891259], unfavorable. All other controls remain in the full report.

| Arm | TRAIN soft easy excess /72 | Development known easy violations /72 | Completion upper violations /72 | Undefined easy risk /72 | Finite support /72 |
|---|---:|---:|---:|---:|---:|
| Unit control |61|58|71|0|1|
| Budget scaled |12|58|67|5|0|

The training optimization measure improves markedly, but development safety does
not pass. The candidate underpredicts selected easy harm in66 of67 defined views,
with median predicted/observed ratio0.040247. Four views select nothing. Unknown
outcomes alone do not explain the58 known-outcome violations. Fewer reported
upper violations alongside undefined views is not a safety improvement claim.

## Mechanism Diagnostic

A separate, frozen144-head TRAIN diagnostic compares the original query-mean
constraint with selected-agent-mass excess, using identical128-query monitor
draws and predictions. For scaled hard actions, easy excess is positive in3/72
query-mean monitors and5/72 agent-mass monitors; two pass only the first. Nine
have undefined easy-reference support. Soft easy excess is positive in12/72
query-mean and14/72 agent-mass monitors. These are known-TRAIN descriptive checks,
not the full TRAIN population or an independent evaluation.

This confirms aggregation non-equivalence but does not identify the cause of
development failure. Soft/hard decisions, missing labels, harm estimation and
recording shift remain separate factors. The intervention establishes that
better empirical constraint optimization, by itself, was insufficient here.

## Reproducibility and Limitations

Native arm64 Torch used four compute threads and zero loader workers. The real
TRAIN pilot verified exact resume/control equality. Training took479.82s and
evaluation223.59s. All payload hashes, splits, input/target alignment, repeated
inference and prior control predictions verify;13,276,438 scalar checks pass.
The new checkpoints total27,870,358bytes and remain outside Git. A TRAIN-only
diagnostic stream interruption resumed with the same code and hashes; no model
or threshold was changed. Aggregate reports can be rechecked without inference.

```bash
.venv-pytorch/bin/python -m scripts.report_m3w_scaled_risk --check
.venv-pytorch/bin/python -m scripts.report_m3w_risk_aggregation --check
```

This is evidence about cost learning, not a new world-dynamics contribution,
physical safety, foundation capability or true3D modeling. The project remains
2.5D and silver labels are not human gold. Independent selection, calibration
and confirmation remain unopened. Deployment, Stage5C and SMC are unchanged.
The existing manuscript AI-use disclosure remains applicable; this addendum
does not replace author responsibility or final submission approval.

[Registered design](protocol.md), [complete readout](readout/conclusions.md),
[failure taxonomy](failure_analysis.md), [TRAIN audit](../european_risk_aggregation_v1/conclusions.md).
