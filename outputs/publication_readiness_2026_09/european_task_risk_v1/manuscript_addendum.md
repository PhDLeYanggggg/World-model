# Decision-Aligned Cost Learning: Development Evidence and Safety Failure

This addendum extends evidence manuscript revision 5 without overwriting it.
All results below come from the registered `european_task_risk_v1` experiment.
They are exposed source-development evidence, not independent confirmation.
The manuscript remains not submission-ready; deployment is unchanged.

## Research Question

Can training a neural cost head for the intervention decision improve utility
over squared-error regression while respecting an unchanged easy-case risk
budget? The forecast producers remain fixed. We test an objective change rather
than a new forecasting architecture or a claim that module combination is novel.

## Matched Design

We train three arms: (i) quadratic regression, (ii) quadratic regression with a
smooth decision-regret term, and (iii) the same task objective with empirical
all-case and easy-case risk dual penalties. Each has 72 matched source/head
identities, giving 216 fresh fits. Initialization, known-TRAIN query sampling,
input packets, width 32, 2,000 updates and the hard deployment gate are fixed.
Head seeds 17, 29 and 43 are not three independently trained forecast producers.

The training surrogate multiplies sigmoids of standardized benefit-minus-harm,
all-risk-budget margin and easy-risk-budget margin, together with causal motion
and TRAIN-support masks. The signed scales use TRAIN only. Temperature decreases
from 1 to 0.1. Duals start at zero, use ascent rate 1 and are capped at 100.
The task weight is 1. Agent losses are averaged within query, then across queries.
Unknown training outcomes are excluded identically in all arms, not imputed.
The empirical penalty is not a finite-sample safety guarantee.

The registered task-risk arm is compared with both contemporary neural controls
and four retained forest controls. Six primary utility contrasts are full and
same-intervention-count comparisons with quadratic, task-utility and original
forest. All six must have positive nominal lower confidence bounds, and the
unchanged completion-support and selected-easy-risk checks must pass. A favorable
weaker comparator cannot compensate for a failed stronger one.

## Data, Units and Uncertainty

The protocol observes 8 steps and predicts 12, with a 12-raw-frame source stride.
Coordinates are dataset-local. Neither physical scale nor elapsed seconds is
established here. Historical raw-frame t+50 is a different task.

The 72 evaluation views span 12 localities and contain 596,988 repeated row
occurrences, including 14,076 unknown-outcome occurrences. Repeated windows,
source views and head seeds are not independent observations. We average views
within locality, then use 3,000 paired-locality bootstrap draws. Intervals are
nominal and not simultaneous or independent confirmatory guarantees.

## Results

Values are task-risk minus comparator, in normalized paired lower-utility
percentage points. Positive values are favorable; they are not FDE reductions.

| Comparator | Full-policy delta [95% interval] | Same-count delta [95% interval] |
|---|---:|---:|
| Quadratic regression | +1.049721 [0.198079, 2.088174] | +0.552791 [0.159252, 0.989134] |
| Task utility only | -1.173144 [-1.628079, -0.724663] | -0.470816 [-0.716578, -0.232627] |
| Original forest | +3.029882 [1.239565, 4.959974] | +0.794163 [0.187919, 1.580280] |

The positive quadratic comparison supports a development decision-utility signal,
including under matched intervention counts. It does not establish that the risk
penalty is beneficial: the task-risk arm loses to its utility-only control.
The remaining forest comparisons and signed-score errors are retained in the
[complete results](readout/conclusions.md), rather than selectively omitted.

## Safety and Failure Mechanisms

The 2% budget measures positive selected easy harm relative to reference error,
not net easy ADE degradation. Missing outcomes retain conservative completion
bounds, and unsupported views do not pass by default.

| Arm | Selected occurrences | Known easy-risk violations /72 | Upper-bound violations /72 | Finite supported views /72 |
|---|---:|---:|---:|---:|
| Quadratic | 107,569 | 60 | 72 | 0 |
| Task utility | 160,275 | 62 | 72 | 0 |
| Task risk | 109,401 | 58 | 71 | 1 |

The candidate therefore fails its registered advancement rule. All 72 task-risk
views underestimate selected easy harm, with median predicted/actual ratio
0.010576. The known-label violations show that missingness alone is insufficient
to explain failure. A lower violation count is not a safety certificate.

A post-readout descriptive check of saved fixed 128-query TRAIN monitors finds
positive easy-risk constraint excess in 61/72 fits and all-risk excess in 38/72.
No dual reaches its cap. The median easy excess is 0.000111244 in normalized cost
units. These monitors are not the entire training distribution or an independent
calibration set. Empirical constraint feasibility has not been achieved even on
most of these fixed monitors; cap saturation is not an observed explanation.

Across evaluation views the mean soft action is 0.108922, while hard intervention
averages 0.283421. The diagnostic soft-at-least-0.5 decision disagrees with the
hard rule by 0.283336 on average. Soft values are not calibrated probabilities.
This suggests surrogate/deployment mismatch but does not isolate it causally.
Constraint scale, gradients and hard/soft agreement should be tested on TRAIN
before registering a repair. More training or rescaling has not yet been shown
to work, and no post-hoc switch to the utility-only policy is made.

## Reproduction and Resource Use

The native arm64 CPU run used four compute threads and zero loader workers.
A real-TRAIN 500-update pilot verified original-control and interrupted/direct
resume equality. Full training took 647.65 seconds, including 57.59 seconds of
TRAIN streaming, and produced 41,752,749 checkpoint bytes. The fixed-final
evaluation took 203.37 seconds. CREATE's free quota was below the preserved
10GiB reserve, so no CREATE training job was submitted; the unchanged experiment
ran locally. This was Torch training, not a NumPy fallback.

All 216 final checkpoints, receipts, inputs and role bindings verify. Repeated
inference and strong-control replay are exact; 11,466,669 scalar checks pass.
The targeted new-path suite passes 56 tests, not a claimed full-repository test.
Private weights and raw inputs remain outside Git. Lightweight receipts, hashes,
aggregate results, protocols and the execution amendments are preserved.

For verification without new training or predictions:

```bash
.venv-pytorch/bin/python -m scripts.report_m3w_task_risk --check
.venv-pytorch/bin/python -m scripts.diagnose_m3w_task_risk_monitor --check
```

The [Chinese runbook](../local_create_runbook_zh.md) records the executed training
commands. Completed experiments must not be silently overwritten or reclassified.

## Limitations and Claim Boundary

This experiment changes cost-head training under fixed forecasters. It does not
demonstrate new neural trajectory dynamics, scene-level joint decision benefit,
independent transfer, foundation-model capability or true 3D modeling. The
current scaffold remains 2.5D, and silver labels are not human gold. Historical
Stage26/37 scores retain their exploratory status; this experiment does not
retroactively make them independent evidence.

Independent selection, calibration and confirmation roles were not opened.
No test threshold search, deployment promotion, Stage5C execution or SMC occurred.
The next experiment must preserve these boundaries and the stronger controls.
The manuscript's existing AI-use disclosure remains applicable to this addendum;
its drafting does not replace author responsibility for every scientific claim.
