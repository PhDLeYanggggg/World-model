# Does Continuing a Turn Improve the Candidate Bank?

I tested continuous turning before fitting another selector. Kinematic
baselines are established controls, not the methodological contribution
(see [Mercat et al.](https://arxiv.org/abs/1908.11472)); this is a mechanism
test on the existing exposed [OpenDD](https://arxiv.org/abs/2007.08463)
development cohort, not an independent leaderboard result.

## Forecast Accuracy Before Selection

All three new predictors estimate a median speed and a de-rotated
tangent from the same three recent displacements. They differ only
in the future heading increments: zero, 0.8-decayed or constant
turn. A stationary chord disables turn extrapolation. No fitting,
threshold search or future input. The turn arms share their first
forecast step. All forecasts use eight past and twelve requested
native steps in dataset-local coordinates.

| Raw predictor | Complete ADE | Available FDE | Paired conservative gain vs OLS4 (pp; 95% CI) |
| --- | ---: | ---: | ---: |
| CV | 0.556 | 1.227 | +6.875 [+3.845, +9.248] |
| OLS4 | 0.620 | 1.319 | reference |
| Shared tangent, straight | 0.582 | 1.260 | +3.741 [+0.292, +7.006] |
| Shared tangent, taper | 0.625 | 1.382 | -8.622 [-23.477, +7.432] |
| Shared tangent, sustain | 0.744 | 1.771 | -31.100 [-50.978, -9.970] |

ADE and FDE are descriptive row-pooled means over 9,823 complete
paths and 9,823 available endpoints out of 10,475 requested
rows. The conservative gain uses the full grid with missing-step
disagreement debits and CV error mass. It averages localities before
bootstrapping 3,000 times. CV is stronger than OLS4 on this roster;
beating OLS4 alone is not sufficient. The sustained turn loses to
the straight control by -34.961 pp [-51.898, -16.630] and its worst
location loses -80.183 pp against OLS4.

## Equal Number, Recipients and Radius of Edits

Add one of the three alternatives at a time to the original six
shapes, projected to the original peak CV offset per edit.
The same past-prefix scoring chooses between the old winner and the
new option, keeping the old winner on a tie. All arms emit the same
1,457 edits on the same rows at the same times and radii. No change
to the original 2% gate. All 21 original location-seed results
and all old short-view witness counts reproduced before comparison.

| Added option | Selected option count | Risk passes | Joint passes / 606 | Direct conservative gain over anchor (pp; 95% CI) |
| --- | ---: | ---: | ---: | ---: |
| None | 0 | 112 | 77 | reference |
| Straight | 159 | 114 | 80 | -0.0207 [-0.0516, -0.0019] |
| Taper | 565 | 110 | 75 | -0.0356 [-0.0752, -0.0046] |
| Sustain | 567 | 111 | 72 | -0.0400 [-0.0838, -0.0057] |

Sustain also loses to the added straight and tapered controls by
-0.0197 pp [-0.0441, -0.0009] and -0.0049 pp [-0.0094, -0.0009].
All arms retain 334 undefined easy-risk views. The 606
recording-seed views represent 202 distinct recordings in seven
locations; the interval averages seeds before resampling locations.

## Exact One-Edit Capacity

For the same 105 initially credit-starved views, the old finite
bank qualifies 30 in hindsight. Adding any one of the three
same-radius shapes gives 39 (the same aggregate count does not
establish the same witnesses). To separate shape from radius and
budget, I replaced the finite shapes with any per-step point inside
the largest legal CV-centered radius. This is an exact calculation
for that relaxed trajectory space, not a deployable predictor.

For a single edit, the edited row must supply a positive easy
reference. At known steps, project the label into the CV-centered
ball to minimize both reference comparisons' prediction error.
Let `C` be the gain over CV at known steps, `O` the whole-recording
gain over OLS4 with missing steps left at CV, and `L` the sum of
reachable distances toward OLS4 at missing steps. Moving a total
`x` toward OLS4 there is optimal for a fixed CV debit, giving
gains `C-x` and `O+x` for `0 <= x <= L`. Strict dual gain is
possible exactly when `C > 0` and `O + min(C,L) > 0`.
The observed error of the edited row never increases, so a
constructed witness satisfies the original risk tests. Every
feasible recording was independently rescored with the original
one-edit scorer. The construction can use the future only because
it is a capacity calculation.

| One-edit option space | Feasible views / 105 |
| --- | ---: |
| Original finite bank | 30 |
| Finite bank plus one matched shape | 39 |
| Any path within the maximum legal radius | 57 |
| Any path with no radius bound | 60 |

These 105 views represent 35 recordings over three repeated seed
views. Removing the radius adds only three seed views, one recording.
Even unrestricted one-edit paths cannot qualify in 45 views, including
the 27 with no eligible positive easy reference in the first-credit
audit. The next targeted test is a matched two-edit comparison that
can separate easy-reference coverage from utility, not another
single-edit amplitude or turn-persistence threshold. Neither this
result nor the two-edit hypothesis changes the approved risk rule.

```sh
.venv-pytorch/bin/python scripts/run_m3w_sustained_turn.py
.venv-pytorch/bin/python scripts/audit_m3w_one_edit_radius.py
```

`fresh_run`: seven-site three-seed readout and exact geometric audit;
four native arm64 Torch threads, zero data workers, no training or
CREATE submission. `cached_verified`: hash-checked OpenDD and frozen
models; byte-identical scored replay and seven resumed site checkpoints.
Both first attempts stopped at historical zero-count fields absent
from JSON; the corrected registrations retain the original attempts.
No deployment, Stage5C or SMC. Independent confirmation is `not_run`;
the 30 DroneCrowd groups remain sealed. No metric or seconds claim.

[Scoring registration](registration.json) | [Scoring results](analysis.json) |
[Capacity registration](radius_ceiling_v2/registration.json) |
[Capacity results](radius_ceiling_v2/analysis.json).
