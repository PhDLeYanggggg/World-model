# AV2: causal references expose only part of the two-edit capacity

The [previous ranking experiment](../av2_assignment_order_v2/README.md)
found 47 exact matched opportunities against one fixed reference on held
development views. I tested whether a better causal reference could create
more meaningful comparisons on the **calibration** role. The held counts
from the previous experiment are not the denominator for this table.

## Design

I reused the same frozen forecast producer, both complete displacement
profiles, ordered recipient menu, CV and OLS4 floors, 8 observed and 12
predicted native steps, validity masks, and original 2% conditions. Alongside
the original and learned additive (soft) references, I evaluated three
causal rules:

| Reference | Choice |
|---|---|
| OLS4 agreement | Minimize total distance of the two edited tracks to the frozen OLS4 rollouts |
| Robust prefix | Reapply each edit to observed-history prefixes, price excess harm, and rank by lower-quartile two-floor proxy gain among options with proxy support |
| Guarded soft | Replace the soft choice only when robust prefix score improves, at least two prefixes support the option, and distance to OLS4 does not increase |

Prefix replay evaluates past observations, not future targets or new
predictions from a shorter context. I froze predictions for the single
calibration city in each rotation before reading its labels. The soft
calibration receipt replays exactly. Scoring reads no held-city outcomes
for that rotation; every rotation belongs to the same development dataset.
No threshold, split, forecast weight, or policy was selected in this
experiment. Seeds 17, 29 and 43 and six city rotations give 340 scenarios,
1,020 seed views, and 465 complete views.

## Calibration-only results

An exact matched opportunity requires the reference and an alternative to
qualify, and the alternative to preserve individual trajectory accuracy and
improve relative-trajectory accuracy after debiting missing steps.

| Reference | Qualifies | Exact matched | Complete-label matched | Known ADE / available FDE | Observed signed easy harm >2% | Conservative gain vs soft (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Original | 57 | 29 | 23 | 0.24960 / 0.55524 | 226 | -1.042 |
| Soft | 65 | 39 | 31 | 0.25203 / 0.56531 | 184 | 0 |
| OLS4 agreement | 70 | 45 | 32 | 0.26095 / 0.58147 | 382 | -5.259 |
| Robust prefix | 89 | 44 | 32 | 0.25633 / 0.57337 | 313 | -1.756 |
| Guarded soft | 82 | 45 | 30 | 0.25337 / 0.56528 | 247 | -0.893 |

Errors are in native coordinates. Guarded soft changes 550 assignments.
It increases the city-weighted matched rate by 0.486 percentage points
(paired city 95% CI [+0.162, +0.816]), but loses 0.893 pp in direct
conservative trajectory gain ([-1.441, -0.339]) and 0.00690 native units in
conservative relative-trajectory gain ([-0.00869, -0.00475]). Its complete
matched count is also lower than soft. A reference that is easier to beat
does not establish an improvement. None of these references is selected.

## Where the capacity goes

316 views contain a standalone qualifying option. If both forecasts
can be chosen with hindsight from the same bank, a fully qualifying
matched comparison exists in **234** views (102 distinct scenarios),
versus **257** before missing-step debits. At least one of the five
causal references exposes that opportunity in only 100 views. For the
frozen soft reference, the 234 views break down as follows:

| Soft reference outcome on a view with some exact matched pair | Views |
|---|---:|
| Soft itself does not qualify | 179 |
| Soft qualifies, but cannot be conservatively improved on both comparisons | 16 |
| Soft qualifies and a matched alternative exists | 39 |

These are outcome-only capacity checks, not release or deployment results.
The shortfall is mainly that the causal reference does not qualify on
views where a different qualifying pair exists. The next experiment
should target the reference's own utility and risk against CV and OLS4
on donor folds, with the frozen soft arm as a control. Do not train
against an intentionally weaker reference to enlarge the measured gain.

## Reproduction

The local freeze took 12.01 s and 617 MB peak RSS on native arm64 with
four Torch threads and no workers. No new training or CREATE job was needed.
The commands require the existing hash-checked calibration assets.
`replay` uses scored receipts without reopening labels.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_causal_reference.py freeze
.venv-pytorch/bin/python scripts/run_m3w_av2_causal_reference.py score
.venv-pytorch/bin/python scripts/run_m3w_av2_causal_reference.py replay
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2_*.py tests/test_m3w_cv_anchor_budget.py tests/test_m3w_cv_joint_feasibility.py
```

To reconstruct the three-way soft-reference breakdown from the local receipts:

```bash
jq -s '[.[].rows[] | select(.any_reference_matched) | .arms.soft]
  | [map(select(.eligible | not)), map(select(.eligible and (.matched | not))),
     map(select(.matched))] | map(length)' \
  data/stage_cvpr2027_experiments/av2_causal_reference_v1/*.score.json
```

[Freeze](freeze.json), [fresh calibration-only results](summary.json).
No official AV2 validation/test or DroneCrowd confirmation data opened.
Manuscript, submission, deployment, Stage5C and SMC unchanged.
