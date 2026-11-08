# Small Support Exchanges at Identical Query Counts

The [fixed-recipient experiment](../incumbent_direction_v2/README.md)
kept 29 undefined recording views. I tested whether replacing a
recipient at the same query can recover support without increasing
displacement. This is an outcome-aware capacity diagnostic on the
same 14 development sources, 34 recordings, three seeds, and native
8/12 steps. It is not an inference policy.

A replacement must be a different row, in the same query, with
observed positive easy reference. Its maximum path displacement
from CV may not exceed that of the removed edit. Of the legal
actions for each new recipient, lowest observed ADE dominates the
others for these constraints; the program uses 185,270 such
alternatives. It either allows one exchange in a recording or at
most one per query. It retains the original selected all/easy
positive-harm tests, full-easy signed degradation test, and positive
reference support. A joint witness must also improve conservative
utility against the incumbent, CV and OLS4 separately, by a fixed
positive numerical margin. No threshold is tuned.

| Budget | Risk-feasible views | Risk rescue among 29 undefined | Joint witnesses | Joint among 29 undefined | Joint among 36 strong-OLS4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| One exchange per recording | 13 | 7 | 3 | 2 | 0 |
| One exchange per query | 31 | 8 | 12 | 2 | 0 |

The risk columns maximize paired utility subject to risk only;
the joint columns additionally impose the three positive lower
bounds. Every returned witness passes the original scorer and
preserves each query count. No solver timed out or failed the
score check. Of the 29 undefined views, none has zero full-roster
easy reference. Twelve have no incumbent edit; five have edits but
no admissible exchange; 12 have an admissible exchange. The 12
per-query joint witnesses come from only three sources. Retaining
the incumbent elsewhere, their all-source conservative utility is
+0.01241 pp, with a 3,000-source-resample 95% interval
[0, +0.02964]. This is an oracle diagnostic on repeated development
views, not independent improvement.

**Why restoring support is not the same as gaining utility.**
If an incumbent edit of radius $r$ is wholly unlabeled, reverting
it to CV debits $r$ in the paired comparison. The observed ADE
gain of the replacement is at most its radius $s$ by the reverse
triangle inequality, even with a partial label mask. The two rows
have the same query weight. If $s \le r$, the paired lower gain of
that exchange is nonpositive, even if it provides the missing easy
reference. Combining exchanges can still help if other exchanges
contribute enough measured gain. A zero-edit recording cannot be
fixed by any count-matched exchange.

I did not train a ranker in this restricted menu. Its outcome-aware
result is too weak to justify using it for confirmation. A next
development comparison must test **when to open a query**, not just
whom to replace: prospectively match the number of opening actions
and their radii between causal controls. Keep the unchanged 2% rule,
including undefined and empty-control handling; do not interpret
the bounds here as limits for a different menu or action budget.
No policy promotion, independent confirmation, Stage5C, SMC, or
new metric/seconds claim. DroneCrowd remains 15 calibration and
30 sealed confirmation groups.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/audit_m3w_small_support_exchange.py
.venv-pytorch/bin/python scripts/audit_m3w_small_support_exchange.py --replay
.venv-pytorch/bin/python scripts/build_m3w_small_support_exchange_replay.py
```

`fresh_run`: exact label-aware capacity under the stated menu, not new
training. Native arm64 CPU, 42 resumable source/seed cells. The
frozen candidates and full replays are `cached_verified`; the
replay independently recomputes each cell before checking it.
`not_run`: CREATE training or independent confirmation.
See [registration](registration.json) and [scalar results](analysis.json).
