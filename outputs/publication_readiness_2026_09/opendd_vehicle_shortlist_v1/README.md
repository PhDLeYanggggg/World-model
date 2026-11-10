# OpenDD vehicles: choose among three frozen angles

I separated the choice of angle from the choice of recording. First I kept
the recordings where the original margin intervened, and changed only the
angle. Then I let the learned heads choose recordings at the same quota.
All arms share the same query, edited vehicle, path family and peak
displacement. The candidates come from the original causal predictions
with eight observed and twelve predicted steps.

I fitted two equally sized heads, one on strict success and one on
success, gain, harm and relative motion. The 126 fits cover 21 excluded
site pairs, three seeds and two heads, at 200 updates each. Nominee
teachers exclude the donor, calibration and held-out sites. The
[pre-score freeze](freeze.json) was pushed as `cb9af83c4`. The unchanged
top-one margin reproduced its prior counts at every operating point.

| Selection | Strict at 526 | Strict at 1,368 | Strict / rotated after calibration |
| --- | ---: | ---: | ---: |
| Original margin, top one | 54 | 107 | 7 / 94 |
| OLS4 agreement, original mask | 50 | 98 | 12 / 191 |
| Strict head, original mask | 55 | 115 | 12 / 167 |
| Utility head, original mask | 56 | 119 | 17 / 205 |
| Strict head, reranked top one | 45 | 109 | 2 / 12 |
| Strict head, reranked top three | 44 | 104 | 2 / 22 |
| Utility head, reranked top one | 50 | 125 | 3 / 25 |
| Utility head, reranked top three | 52 | 126 | 13 / 90 |
| Hindsight top three, original mask | 85 | 194 | Not a policy |

The 526 and 1,368 columns match intervention counts in each direction.
The calibration column uses independent site thresholds and is not a
matched-count comparison. Strict means the original risk test and both
reference checks pass, the conservative gain against the unrotated
prediction is nonnegative within numerical tolerance, and relative motion
improves. Positive incremental utility alone is a separate column in the
[readout](readout.json). The calibrated utility top-three policy passes
risk on 49 of 90 rotations, and 16 pass risk, both references and
positive incremental utility. It does not replace the original policy.

| Utility fixed minus margin | Conservative trajectory gain (pp) | Available FDE gain (pp) | Relative-motion gain (native units) |
| --- | ---: | ---: | ---: |
| Same 526 rotations | -0.00004382 [-0.00010071, +0.00000052] | -0.00003447 [-0.00008906, +0.00000154] | -0.00024987 [-0.00047145, -0.00004879] |
| Same 1,368 rotations | +0.00008106 [+0.00005051, +0.00010999] | +0.00004559 [+0.00002039, +0.00007312] | -0.00064061 [-0.00115208, -0.00025968] |

Positive means lower error. Trajectory and available FDE gains use the
corresponding CV error denominator. The conservative contrast is a direct
pairwise lower bound, not a subtraction of two lower bounds. Intervals
use 3,000 paired draws over seven locations after aggregation over
directions and seeds. There are 8,874 repeated views of 493 recordings;
views are not independent. These are OpenDD development comparisons,
not independent confirmation.

The three angles contain a strict solution in 802 views versus 420 for
the first angle. But that increase combines the ability to find new
recordings with the ability to improve the original actions. On the
original 526 masks, only 31 additional views are recoverable even in
hindsight. The learned utility model recovers just two net strict
successes, and loses relative-motion utility. Reranking recordings
does not fix it: at 526 rotations the utility top-three rule scores
52 strict successes versus the original 54. A quota-dependent trajectory
gain is not evidence of better joint dynamics or safe release. The next
useful development contrast is to preserve the original mask and test
whether choosing only when the predicted trajectory and relative-motion
improvements agree can retain the gain without moving along this tradeoff.

## Replay

Run from the repository root using native Apple Silicon Python, with
four Torch threads and no DataLoader workers:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_vehicle_shortlist.py freeze
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_vehicle_shortlist.py score
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_vehicle_shortlist_fixed_mask.py
```

The first command checks or resumes local checkpoints and frozen score
packets; the second reuses completed direction receipts and verifies the
unchanged comparator. The third reads the same OpenDD development
outcomes to compute the hindsight bound. No external outcome was opened.
