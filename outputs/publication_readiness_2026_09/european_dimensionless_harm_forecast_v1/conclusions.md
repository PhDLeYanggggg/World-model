# Reference-Harm Forecast Comparison

29 October 2026. Fresh neural training and source-development readout, with
cached-verified controls. **No deployment promotion; not submission-ready.**

## What Was Tested

The loss adds two positive-ADE-harm terms, with fixed coefficients of 2, against
the training-selected motion baseline and causal constant velocity (CV). Data,
folds, seeds, architecture, sampling, initialization, optimizer and 4,000-update
schedule stay fixed within each model family. Future labels are supervision,
never inference inputs. This is not another threshold search.

Two families were run and must remain separate:

| Family | Primary fits / updates | Training time | Peak RSS | Role |
|---|---:|---:|---:|---|
| Actual current producer: dimensionless agent-track model | 9 / 36,000 | 675.416 s | 1.633 GB | Direct loss-only repair comparison |
| Earlier flat complete-neighbor forecaster | 18 / 72,000 | 1,640.711 s | 1.734 GB | Separate historical-family comparison |

Both completed with observed exit code 0. Each also completed an 80-update
real-data engineering pilot and an exact 4,000-update neutral replay. Interrupted
10+10 updates exactly match 20 continuous updates. Neutral training reproduces
the control weights, optimizer, RNG state and draws, not just a similar loss.
All endpoints were committed before their first readout: current family in
`d2f8eda7`, earlier family in `41f0d3f0`. No seed or checkpoint was selected.

The initial lineage assumption was corrected before reading new results. The
earlier family is not the current policy's producer. Its comparison cannot be
presented as evidence that the current policy was repaired. See the preserved
[lineage clarification](../european_reference_harm_forecast_v1/lineage_clarification.md).

## Current-Producer Result

Positive numbers below mean improvement over causal CV. These are equal-locality
averages, after averaging repeated producer/seed errors within locality.

| Endpoint | Original (%) | Harm-aware loss (%) | Change (pp) |
|---|---:|---:|---:|
| All ADE | 8.461640 | 7.531916 | -0.929724 |
| Complete-future ADE | 12.305600 | 11.053453 | -1.252148 |
| Easy ADE | -11.169208 | -8.433059 | +2.736149 |
| Hard ADE | 13.355213 | 11.675472 | -1.679741 |
| All FDE | 13.398345 | 11.588928 | -1.809418 |

The 3,000-draw paired locality-bootstrap interval for the all-ADE change is
**[-1.669310, -0.168497] pp**. All three seeds regress: -0.898220, -0.667846
and -1.223106 pp. Four localities improve and eight regress. Easy error improves
relative to the old neural model, but its 8.43% degradation against CV is still
above 2%. This is an accuracy/safety tradeoff, not a successful repair.

Positive harm tells a different story from net error. Equal-locality known
positive ADE harm falls from 13.385360% to 11.756180% against CV, and from
6.493343% to 2.472366% against the training-selected baseline. The reference
matters: the latter is damping .90 in folds 0/1 and history OLS4 in fold 2,
not the frozen CV/damping .97 controller used by the policy.

The unrestricted candidate still has 963/978 recording/producer/seed views
with known all-harm above 2% against CV, and 922/978 on easy rows. The 978
views repeat 163 recordings; they are not 978 independent recordings. Every
all-row recording view contains some unknown future labels. These diagnostics
cannot certify the complete intervention policy.

## Exact Fixed-Floor Support

The original label-aware TRAIN oracle was reproduced on all 148 required
recording/context views, covering 74 unique source-qualified recordings.
The original CV/damping .97 floor, eligibility and 2% budgets were retained.
Replacing only the candidate forecasts changes feasible views **146 to 147**.
One view gains support and none loses it.

Locality 082, recording 51 remains unsupported in the
`single1_seed43_controller2_dimensionless` view. Its best eligible easy
individual harm/reference ratio falls from 2.973595 to 1.931046, still far
above the permitted ratio 0.02. The other context for that same recording
gains a zero-harm easy anchor. This is a limited support improvement, not a
learned safe policy or independent generalization result.

## Output-Bound Diagnosis

The current producer bounds its correction around its training-selected
baseline. A post-fit, past-only geometry check finds that CV is outside that
bound on **43,743 of 140,585 excluded-source rows** for fold 2. The same rows
repeat across its three seeds; they must not be counted three times. Folds
0/1 can geometrically represent CV on every evaluated row.

For both problematic 082/51 TRAIN views, CV is inside the output bound on all
78 rows. A label-aware independent-step ball relaxation is much better than
the actual candidate there, but this oracle is not a realizable neural model
or proof that causal context identifies the future. The remaining recording
failure is therefore not explained by the hard output radius alone.

This separates two issues: some rows have an output-coverage restriction;
the known difficult recording also has a learning/context/capacity problem.

## Earlier-Family Result

The separate earlier forecaster also fails to show a reliable gain. Its
equal-locality ADE improvement over CV falls from 2.157596% to 1.389049%.
The paired change is -0.768547 pp, with conditional interval
[-2.281846, +0.832023] pp. This interval spans zero; it is not evidence of
stable improvement or a definitive general inferiority claim.

Its equal-locality easy ADE degradation remains 13.552605%, versus 13.924449%
for its control. The pooled easy-risk columns in that family's compact
readout are separate descriptive aggregates; they must not replace the
equal-locality primary estimator or stand in for complete-policy safety.

## Evidence Boundaries

The population is 318,969 windows, 163 recordings and 12 already-opened European
SOURCE localities. Of these, 311,922 rows have at least one future label,
193,705 have complete requested futures and 7,047 have no future labels.
The task is 8 observations and 12 requested predictions at stride 12 raw
frames. Coordinates are image pixels and labels are released detector tracks,
not human gold. No metric, seconds-level, true-3D or foundation claim follows.

The readouts replay exactly. The current family reproduces 864 original
control metric values; the earlier family reproduces its original primary
metrics. The 148-view support diagnostic and post-fit analyses also replay
exactly. There are 54 passing targeted regression tests; the complete historical
test suite was not rerun. These are execution checks, not scientific success.

Independent model-selection, risk-calibration and confirmation roles remain
unopened. No new candidate-dependent policy head or independent safety
calibration was trained in this comparison. Historical failures are retained;
empty matched risk is not_estimable, and count matching is utility only.
The complete policy still requires the original 2% criteria on every required
scene/recording. Stage5C and SMC remain off.

## Next Controlled Change

The shortest useful follow-up is a reference-coverage ablation, not a larger
network or a new threshold grid. Keep this trained loss comparison and the old
producer frozen. Define a causal output construction that explicitly includes
CV as well as the training-selected baseline, then compare it under the same
training and source-development budget. Change that construction alone first;
do not simultaneously tune loss weights, split, thresholds and architecture.

After any forecast change, retrain candidate-dependent benefit/harm heads on
training data. A source-development improvement would still need independent
calibration and confirmation before deployment. The 082/51 context limitation
needs separate TRAIN-only analysis rather than excluding that recording.

The follow-up is **not_run**. This iteration is complete; the M3W research goal
and submission-evidence work are not complete.
