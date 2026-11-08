# Optional query allocation with two reference risks

**Decision: no deployment change.** Joint choice reduces measured risk
at the same intervention counts, but it loses conservative utility to
the matched simple control and does not qualify for calibration.

## Matched development experiment

The roster is the original 24 MovingDroneCrowd++ fitting recordings
from nine groups (69,746 rows, 1,176 queries; 8 observed / 12 predicted
steps). Source groups are split 6/2/1 for fitting, inner calibration and
outer scoring. The head estimates reference masses, positive harms and
support on the same ten quarter-strength causal paths. Missing-support
probability and path disagreement give the two utility increments.
Future outcomes and label masks are supervision, never inference inputs.

Each query has a cap of the smaller of two and the inherited past-only
count. This reduces the maximum from 31,923 to 2,173 changes per seed.
Joint search considers identity, one change, and two different recipients
over a deterministic shortlist sorted by gain, price-adjusted gain and
both individual risk channels. The independent gain and price-adjusted
controls see the whole same path bank and copy only the joint count per
query. The forced control fills the two-person cap where legal. The
deployment scorer is unchanged; relaxed *predicted* excess allowances
are candidate model settings, not a change to the 2% evaluation rule.

| Held development policy | Changes / 209,238 rows | Risk passes / 72 | All-risk over 2% | Undefined easy risk |
|---|---:|---:|---:|---:|
| Forced cap, gain ranking | 6,519 | 2 | 64 | 15 |
| Strict joint | 1,765 | 24 | 0 | 39 |
| Same-count gain ranking | 1,765 | 0 | 46 | 37 |
| Same-count price ranking | 1,765 | 12 | 20 | 38 |

The primary direct contrast, strict joint minus same-count price, is
**-0.0716 pp** conservative gain (95% paired source interval
[-0.1224, -0.0271]). Strict joint versus CV is -0.00015 pp
[-0.00118, +0.00069]; versus OLS4 it is -12.128 pp
[-21.779, -2.463]. Three thousand source bootstrap draws average
the three actual training seeds before resampling groups. Across the
pre-fixed predicted-slack ladder, the direct joint-minus-price utility
stays negative and no option qualifies on any of the 27 inner
fold-seeds. The calibrated arm falls back to CV. The 39 undefined
easy views are not risk passes.

## What the action set can deliver

With future labels, constrained optimization over the same quarter
paths can satisfy the original rule and show positive conservative gain
against both references in **3/24** recordings, whether it must fill
the two-person cap or may make fewer edits. Both arms have 17 negative
objective bounds and four risk-infeasible recordings, with none
unresolved. This comparison keeps the new cap fixed; it is not a
comparison against the larger inherited count.

The second diagnostic ignores risk and the CV requirement altogether
and optimizes the conservative OLS4 score exactly. At each known row,
it picks the lowest available-step ADE path; for a wholly unknown row
it picks the path closest to OLS4 in peak distance. Sorting the
nonnegative gains within the query then gives the exact additive
upper score under the cap. A stronger relaxation preserves waypoint
radii but lets every direction vary independently with labels.

| Maximum changes per query (also capped by the inherited count) | 2 | 4 | 8 | 16 | Inherited |
|---|---:|---:|---:|---:|---:|
| Quarter paths: recordings with strictly positive OLS4 upper score | 5 | 5 | 5 | 7 | 9 |
| Arbitrary waypoint directions: strictly positive | 5 | 7 | 9 | 11 | 15 |

At the two-person cap, **19/24** remain strictly negative even in
the direction relaxation. Raising only the count still leaves nine
strictly negative at inherited caps. This limits the tested
quarter-path family on the conservative score, not the ability of
other forecasters to beat OLS4. A useful next test must address the
forecast family and not just change the cutoff or fill more slots.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_optional_allocation.py
.venv-pytorch/bin/python scripts/run_m3w_optional_allocation.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_optional_allocation_capacity.py
.venv-pytorch/bin/python scripts/audit_m3w_optional_allocation_ols_limit.py
```

Native arm64 Torch, four threads, zero loader workers; 27 fits
(seeds 17, 29, 43; 500 updates per fit), 55.76 s fresh and 36.39 s
checkpoint-only replay. The v1 input assertion stopped before fitting
because the inherited count can reach 184. v2 explicitly freezes the
two-person limit and all controls use it. Forty targeted tests passed.
No DroneCrowd calibration or confirmation was opened, no Stage5C
or SMC was run, and these are development results in image coordinates.

[Registered design](registration.json) | [Paired analysis](analysis.json) |
[Dual-reference capacity](capacity.json) | [OLS4 ceiling](ols_limit.json) |
[Supplement v47](../evidence_manuscript_v47/README.md).
