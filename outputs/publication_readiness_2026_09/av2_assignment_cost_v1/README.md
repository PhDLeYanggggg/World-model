# AV2: learned assignment costs at a fixed edit budget

Does learning relative gains and harms improve assignment of the same two
displacement profiles? Does a learned interaction term help once both arms
share the same additive estimates? The first helps recover more standalone
opportunities in development, but neither arm is reliable enough for release.
The second worsens the paired trajectory and relative-position contrasts.

## Comparison

The candidate bank, two full per-step displacement profiles, 8 observed and
12 predicted native steps, CV/OLS4 comparisons, missing-label debits and
original 2% rules are unchanged from the
[reassignment study](../av2_recipient_reassignment_v1/README.md). Each of six
development-city rotations uses four donor cities and a separate calibration
city. I trained 18 fits (seeds 17, 29 and 43; 400 optimizer updates each) and
froze the held predictions and calibration thresholds before scoring. There
are 340 scenarios and 1,020 seed views, 465 with complete labels. Seed views
and windows are not independent samples.

The additive arm learns each actor's easy support, gains relative to both
floors, excess positive harm and signed easy change. The joint arm adds one
learned nonadditive error reduction on the relative trajectory of the two
edited agents. It otherwise shares the additive arm's costs and candidates.
Forecasts use only observed history, retrospective prefix replay and causal
motion predictions. All future labels used for fitting are donor labels.

## Scoring results

| Forced assignment | Qualifies | Complete-label qualifies | ADE | FDE (available endpoints) | Worst view p95 ADE | Conservative gain vs original (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Original assignment | 63 | 43 | 0.2480 | 0.5518 | 5.929 | 0.000 |
| History support | 66 | 41 | 0.2580 | 0.5764 | 7.390 | -4.316 |
| History support + pair | 72 | 44 | 0.2550 | 0.5696 | 7.390 | -3.492 |
| Learned additive | 89 | 57 | 0.2509 | 0.5640 | 7.390 | -2.048 |
| Learned joint | 80 | 56 | 0.2513 | 0.5649 | 7.432 | -2.147 |

Errors are in native coordinates; the table does not claim physical safety or
convert horizons to seconds. Paired city resampling gives joint minus additive
qualification of -1.179 pp (95% CI [-2.482, 0.189]), direct conservative
trajectory gain of -0.356 pp ([-0.695, -0.021]), and relative-trajectory gain
of -0.00212 in native coordinates ([-0.00349, -0.00080]). These intervals
describe this development cohort, not independent confirmation.

The original, learned additive and learned joint forced assignments have
382, 387 and 386 views with a defined easy risk bound, respectively.
Their observed signed easy degradation exceeds 2% in 210, 171 and 172
views, respectively. Undefined support is not a safety pass.

The standalone additive and joint thresholds each selected two held views,
of which only one passed. Forced joint assignment beat additive under the
full matched criterion in four views. The calibrated strict matched policy
selected zero held views; it did not demonstrate safety or utility.

## What is missing

The following substitutions use held outcomes only after scoring. Each
column is the count of exact eligibility after replacing one or more
predicted quantities while leaving the candidate bank and rank rule fixed.
These are diagnostic counts, not candidate policies.

| Substitution | Additive | Joint |
|---|---:|---:|
| None | 89 | 80 |
| Known easy support | 132 | 130 |
| Known risk and signed easy change | 164 | 150 |
| Known relative utility | 174 | 166 |
| Known pair residual | 89 | 93 |
| Known support, risk and utility | 307 | 299 |
| All terms known | 307 | 309 |
| Any exactly qualifying assignment in the bank | 317 | 317 |

Of the additive arm's 228 missed opportunities, 121 lacked easy support in
the chosen assignment; the other 107 had support but failed a risk or gain
test. Knowing the pair term alone recovers less than knowing risk or utility.
Even with all terms known, the soft rank rule leaves some legal alternatives
unchosen. Next compare donor-trained, separately constrained support, risk and
utility decisions against the additive arm before spending on a larger pair
model. Retain the same candidates, no-change control and calibration rule.

## Reproduction

Use the hash-verified [parent experiment](../av2_recipient_reassignment_v1/README.md)
and its local development inputs. On Apple Silicon, use native arm64 Python,
four PyTorch threads, one interop thread and no DataLoader workers. Running
`fit` again resumes from a checkpoint (saved every 50 steps) and checks the
frozen roster. `replay` reproduces the summary from scored receipts without
reading held labels. The attribution script requires the already scored
receipts and does read held labels. No additional CREATE job was needed.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_assignment_cost.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_assignment_cost.py score
.venv-pytorch/bin/python scripts/run_m3w_av2_assignment_cost.py replay
.venv-pytorch/bin/python scripts/audit_m3w_av2_assignment_cost_components.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2_*.py tests/test_m3w_cv_anchor_budget.py tests/test_m3w_cv_joint_feasibility.py
```

Results and substitutions: `fresh_run`; matched controls reproduced from the
frozen parent bank. [Metrics](summary.json) and [attribution](component_attribution.json).
No AV2 official validation/test or DroneCrowd independent confirmation
outcomes were opened. No deployment, manuscript or submission change.
Stage5C and SMC remain disabled.
