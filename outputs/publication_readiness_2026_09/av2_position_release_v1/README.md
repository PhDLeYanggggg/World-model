# AV2: the frozen position projection does not pass release

The [held development comparison](../av2_relative_projection_v1/README.md)
found more qualifying forecasts from the frozen position projection than
from the additive reference (119 versus 89). Here I test whether a causal
confidence score can identify those forecasts reliably enough to release
them under the unchanged 2% rules. The predictor itself is not refit.

## Matched policies

Every score sees the same 8 observed and 12 predicted native steps, the
same forecasts, exactly the same two recipients and per-step displacement
budget, and the same validity masks and evaluation. The intervention is the
frozen position path; nonreleased predictions are unchanged CV.

- **Trajectory replay:** transport the two frozen edit shapes to four
  observed-history prefixes and take the lower-quartile proxy gain against
  CV and OLS4.
- **Risk replay:** use the same gain, subtract positive excess from all,
  easy and signed easy risk proxies, and downweight low edited-easy support.
- **Transferred cost:** apply the existing donor-only action cost heads
  to the projected paths. These heads learned from the transported bank,
  not from new projection labels.

Each score uses the original largest nonempty all-qualified confidence
prefix on the separate calibration city. Ties cannot be split. The
registered combined policy chooses the score that releases the most
qualifying calibration views, with a fixed tie order. No score, threshold,
model, or grouping is chosen on held outcomes. There are six city
rotations and seeds 17, 29 and 43. The 1,020 seed views are 340 development
scenarios, not 1,020 independent samples.

## Calibration and held readout

The frozen projection has 105 qualifying calibration views and 119
qualifying held development views before applying any confidence gate.

| Policy | Qualifying calibration prefix views | Held released / qualifies | Held risk fails | Held signed easy >2% | Held dual-floor fails |
|---|---:|---:|---:|---:|---:|
| Trajectory replay | 0 | 0 / 0 | Not estimable | Not estimable | Not estimable |
| Risk replay | 9 | 26 / 6 | 18 | 11 | 11 |
| Transferred cost | 6 | 6 / 3 | 3 | 1 | 1 |
| Calibration-selected | 12 | 26 / 6 | 18 | 11 | 11 |

The combined policy chose risk replay in five folds, transferred cost in
three, and abstained in ten. It releases 26 views (2.55%; 25 distinct
scenarios), the same set as risk replay on this held cohort. The full
policy's conservative CV gain is -0.0127 pp (paired city 95% CI
[-0.0990, +0.0516]). Risk and gain failures can overlap. The fact that
the transferred score's six releases have positive mean same-set gain
against both floors does not compensate for three failed risk checks.
There is no deployable policy here.

| Nonempty held subset | Known ADE / available FDE | Worst released p95 ADE | Same-set conservative gain vs CV / OLS4 (pp) |
|---|---:|---:|---:|
| Risk replay / calibration-selected | 0.17962 / 0.40298 | 2.182 | -0.496 / +6.374 |
| Transferred cost | 0.06229 / 0.16107 | 0.658 | +3.742 / +12.827 |

Errors are in native units. These are different subsets, so the ADE/FDE
numbers are not a matched comparison between policies. Abstention is
not an estimated safety rate on an empty released subset.

## Why release fails

All 18 calibration folds contain at least one qualifying forecast.
The highest score group qualifies in zero folds with trajectory replay,
five with risk replay, and six with transferred cost. On held views,
mean fold average precision is 0.155, 0.347 and 0.443, respectively,
against a 0.122 mean positive rate. Their top-ranked choices qualify in
0, 3 and 6 of 18 folds, respectively. This rank audit is diagnostic only.

Risk information helps find candidates, but the upper confidence tail
is not reliable enough to transfer the original release rule. This is
now a decision problem on a usable fixed predictor: the next experiment
should fit a projection-specific risk and gain estimator with
out-of-source forecast lineage and compare it at the same intervention
counts. There is no reason to fit another angle head before that check.

## Reproduction

The confidence freeze took 14.19 s and 649 MB peak RSS on native arm64,
four Torch threads and zero loader workers. It reads no outcome receipts.
Scoring uses the old sealed scalar receipts, verifies their role and row
alignment, freezes calibration before reading held receipts, and opens
no raw future labels. The audit verifies that the parent metrics replay
and records digests for every scored receipt. No new training or CREATE
job was needed.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_position_release.py freeze
.venv-pytorch/bin/python scripts/run_m3w_av2_position_release.py score
.venv-pytorch/bin/python scripts/audit_m3w_av2_position_release.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2_*.py tests/test_m3w_cv_anchor_budget.py tests/test_m3w_cv_joint_feasibility.py
```

[Freeze](freeze.json), [calibration](calibration.json), [scores](summary.json),
[rank audit](confidence_attribution.json), [outcome lineage](outcome_lineage.json).
No AV2 official validation/test, DroneCrowd independent confirmation,
manuscript, submission, deployment, Stage5C, or SMC change.
