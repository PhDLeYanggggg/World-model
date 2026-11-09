# AV2: separating assignment constraints did not improve release

The [previous component audit](../av2_assignment_cost_v1/README.md) showed
that better utility and risk estimates could recover more assignments than
a better pair residual. I tested whether enforcing these estimates as separate
conditions would improve decisions on the same candidates.

## Matched comparison

All three policies have the same two transported per-step displacement
profiles, candidate actors, 8 observed and 12 predicted native steps,
label masks, CV and OLS4 floors, and original 2% evaluation.

| Policy | Candidate decision |
|---|---|
| Soft cost | Exact replay of the previous learned additive cost and score |
| Analytic constraints | Require predicted easy support >= 0.5, nonpositive risk excesses and signed easy excess, and positive gain vs both floors |
| Learned constraints | Require the product of three learned conditional probabilities to be >= 0.5: support, risk given support, and utility given support and risk |

Both new policies rank their admissible candidates by the same frozen
minimum predicted gain over the two floors. When none is admissible,
their forced-choice diagnostics use the original soft choice, but
their actual release policy abstains to CV. The confidence score is the
predicted support for the analytic rule and the joint probability for the
learned rule. The calibration city selects the original nonempty
all-qualified prefix for each arm independently.

Four donor cities fit each model. A fifth calibrates the policy; a sixth
is held out for scoring. The learned arm uses conditional binary losses
with absent antecedents masked, and target conjunctions are checked against
the exact partial-label evaluator. Candidate history, including relative
motion between the two actors, is past-only. Seeds 17, 29 and 43 across six
rotations produced 18 fits and 7,200 updates. Local arm64 CPU training and
freezing took 39.08 s with 897 MB peak RSS, four Torch threads and no loader
workers. The checkpoint stores model, optimizer, and sampling RNG every
50 steps. No additional CREATE job was needed.

## Development results

There are 340 scenarios, 1,020 seed views, 465 complete-label views, and
56,580 ordered options. A total of 317 views have at least one exactly
qualifying option. These are development outcomes, not independent tests.

| Policy | Views proposing before calibration | Forced qualifies | Calibrated held selections / passes | Forced ADE / FDE | Worst view p95 ADE | Observed signed easy degradation >2% |
|---|---:|---:|---:|---:|---:|---:|
| Soft cost | 1,020 | 89 | 2 / 1 | 0.25091 / 0.56398 | 7.390 | 171 |
| Analytic constraints | 47 | 89 | 8 / 1 | 0.25100 / 0.56414 | 7.390 | 169 |
| Learned constraints | 0 | 89 (soft fallback) | 0 / 0 | 0.25091 / 0.56398 (soft fallback) | 7.390 | 171 (soft fallback) |

Distances are in native coordinates. Analytic minus soft forced
conservative gain is -0.041 percentage points (paired city 95% CI
[-0.105, +0.008]); relative-trajectory gain is -0.000287 native coordinate
units ([-0.000742, -0.000013]). All eight analytic releases came from
Pittsburgh. At exactly those eight views, the original soft assignments
also pass once, and the analytic choice's conservative gain versus
soft is never positive (six ties and two losses). No deployment change.

## Why the learned rule abstains

Only 2,494/56,580 options (4.41%) meet all three conditions. The estimated
mean joint probability is 4.25%, and the median of the within-view maximum
is 9.29%; the highest option reaches 42.55%, below the fixed 50% rule.
The conditional risk pass rate after known support is only 15.11%.
The fitted utility score has Brier 0.273 on options with known support
and safe risk, worse than the 0.239 hindsight constant-rate reference
on that same subset. This reference is not a fitted policy.

Removing the cutoff only for an outcome-only ranking audit does not
explain away the failure: the learned probability's top candidate qualifies
in 44 views, vs 89 for the soft cost. The paired city difference is -4.489
percentage points (95% CI [-5.966, -2.956]). Top-3, top-5, and top-10
contain a qualifying option in 112, 157, and 205 views, respectively.
This is an audit, not a selected top-k policy or a threshold change.
The next model should learn ordering within each scenario from continuous
risk and utility margins and compare against the same soft cost before
calibration, instead of searching another cutoff for these scores.

## Relation to calibration literature

[Conformal Risk Control](https://arxiv.org/abs/2208.02814) treats expected
monotone loss under its sampling conditions; [Learn then Test](https://doi.org/10.1214/24-AOAS1998)
uses multiple testing to calibrate policies with finite-sample guarantees.
The all-qualified city prefix used here is an empirical release rule, not
an implementation of either guarantee. The research question here is whether
assigning the same two edits improves baseline-relative utility while
preserving supported risk conditions; the 2% harm cap is not a 2% rate of
unsafe releases.

## Reproduction

The original frozen cost study and its hash-checked local input assets are
required. Re-running `fit` resumes saved checkpoints and verifies existing
receipts. `replay` reads scored receipts without reopening labels; the
ranking audit separately uses only already scored development outcomes.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_assignment_constraints.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_assignment_constraints.py score
.venv-pytorch/bin/python scripts/run_m3w_av2_assignment_constraints.py replay
.venv-pytorch/bin/python scripts/audit_m3w_av2_assignment_constraint_ranking.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2_*.py tests/test_m3w_cv_anchor_budget.py tests/test_m3w_cv_joint_feasibility.py
```

[Fresh scoring](summary.json), [frozen ranking audit](frozen_ranking_diagnostic.json),
and [freeze receipt](freeze.json). No AV2 official validation/test or
DroneCrowd independent confirmation outcomes opened. Manuscript,
deployment, Stage5C, and SMC unchanged.
