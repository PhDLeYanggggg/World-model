# AV2: Harm Coverage Drops When the Same Estimate Selects the Edit

I revisited the [scene-margin assignment](../av2_margin_assignment_v2/README.md)
after its standalone release failed 15 of 16 held checks.
The question was whether a separately fitted upper harm
estimate fixes that optimism, and whether its coverage
survives when it also decides which edit is taken.

## Frozen Model Audit

The prior model's predictions and thresholds replay exactly.
It predicts nonpositive all-positive-harm excess in all 1,020
held two-slot views, although 640 actually exceed the allowance.
Among its 16 held releases, nine fail the all-positive-harm
check, ten fail the easy-positive-harm check, seven fail
signed easy preservation, and six have no defined edited-easy
reference. Those counts overlap. One view qualifies. The
prior model's margin logits should not be interpreted as
calibrated harm estimates.
[Exact audit by fold](../av2_margin_assignment_v2/harm_margin_audit.json).

## Matched Experiment

I freeze the previous CV/OLS4 gain estimates and easy-support
head. The new network predicts all-positive, easy-positive
and signed-easy excess on the same two-edit menu. One head
fits scene harm with Huber loss. A second learns a
nonnegative increment using [quantile loss](https://doi.org/10.2307/1913643)
at 0.8, with the point estimate detached. Losses are applied
to the whole assignment, not to separate actor quantiles
that are then assumed to sum to a valid bound.

Four readouts share the candidate paths and budget: the prior
score; the new point harm score; the new upper harm score
(primary); and the upper harm score used only for scene order,
with the prior choices left unchanged. Training samples only
donor scenes and their existing city-excluded options. Six
city folds, seeds 17/29/43, 896 held development scenarios,
8 observed/12 forecast steps, and the original 2% and dual
motion-floor checks are unchanged. No held labels enter
training, threshold selection or forecast freezing.

## Coverage Depends on Selection

| Harm excess | Upper coverage on frozen choices | Upper coverage after upper-score selection | Paired change in coverage, pp [95% CI] |
|---|---:|---:|---:|
| All positive harm | 77.94% | 70.00% | -6.77 [-11.96, -0.89] |
| Easy positive harm | 77.06% | 70.39% | -5.57 [-10.07, -2.23] |
| Signed easy | 82.75% | 70.29% | -11.17 [-15.14, -6.75] |

The first two columns pool 1,020 seed views of 340 scenarios.
The last column averages seeds within scenario and then uses
3,000 paired city bootstrap draws. The same upper estimator
and the same scenes enter both sides, with different chosen
paths in 878 views. The measured-minus-predicted harm
residual increases for all three terms after selection.
The pattern is consistent with the
[optimizer's curse](https://doi.org/10.1287/mnsc.1050.0451):
choosing among estimated options changes the errors of the
chosen option. The observed effect is on this development
menu, not a distribution-free bound.

On the frozen paths, false-safe signs for overall harm fall
from 129 with the new point estimator to 12 with the upper
one. For easy-positive and signed-easy harm they fall from
124 to 39 and 107 to 23. After selection by the upper score,
the corresponding upper-estimate counts are 36, 50 and 51.
This test distinguishes improvement in harm estimation from
improvement in the actual release decision.

## Forecasts and Release

| Forced reference | Known ADE | Available FDE | Conservative gain vs CV (pp) | Risk passes / 1,932 | Signed easy violations | Eligible / 1,020 |
|---|---:|---:|---:|---:|---:|---:|
| CV | 0.24668 | 0.57458 | 0 | No edits | No edits | Not applicable |
| Fixed independent control | 0.24574 | 0.56860 | -1.07 | 143 | 242 | 63 |
| Scene uniform | 0.24631 | 0.56954 | -1.24 | 132 | 250 | — |
| Prior margin model / frozen upper | 0.24634 | 0.56991 | -1.43 | 144 | 354 | 61 |
| Point harm | 0.24695 | 0.57067 | -1.64 | 145 | 350 | 61 |
| Upper harm | 0.24585 | 0.56852 | -1.18 | 152 | 268 | 75 |

All forced controls except CV use 2,952 edits. The upper
arm gains +1.60 pp in city-balanced qualification versus
the prior margin model, with interval [-0.53, +3.55].
Its hard gain versus CV is +0.47%, versus +0.56% for the
fixed independent control. Its direct full-grid pair gain
against that fixed control is -0.00365 native units
[-0.00465, -0.00265]. Neither point nor upper passes any
of 1,020 nonempty calibration prefixes. The frozen-choice
upper readout passes one prefix, from Dearborn with Austin
held, but its one held release fails original risk.
There is no joint-policy result or deployment change.

The next useful test is to train harm on choices actually
made by a city-excluded ranker, rather than only on uniformly
sampled legal options. Compare a separately fitted veto
against using the same estimate for optimization and release.
This targets the measured coverage drop without changing
the risk allowance or evaluating reserved outcomes.

## Reproduce

Requires the [margin assignment assets](../av2_margin_assignment_v2/README.md).

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/audit_m3w_av2_harm_margins.py
.venv-pytorch/bin/python scripts/run_m3w_av2_harm_quantile.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_harm_quantile.py score
.venv-pytorch/bin/python scripts/audit_m3w_av2_harm_postselection.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2*.py tests/test_m3w_interaction_frontier.py tests/test_m3w_slate_relative_gain.py
```

18 fits on native arm64 PyTorch, four compute threads and
zero loader workers. Fit/freeze took 41.62 s (771 MB peak RSS);
score 10.23 s, initial audit 14.13 s, paired audit 4.56 s.
Checkpoints every 40 steps retain optimizer and sampler state.
Completed receipt replay took 7.20 s without new fits.
54 focused tests passed. No CREATE job, manuscript edit,
independent confirmation access, Stage5C, or SMC.

[Registration](registration.json) · [Freeze](freeze.json) ·
[Results](summary.json) · [Paired coverage](postselection.json)
