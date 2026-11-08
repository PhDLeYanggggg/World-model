# Does a Later Second Edit Improve the Recording?

This is the causal follow-up to the [two-edit capacity audit](../two_edit_v1/README.md).
I reuse its 21 frozen value heads, seven OpenDD development locations,
candidate bank, CV and OLS4 references, eight observed and twelve requested
native steps, missing-step debits and original 2% rule. No retraining,
threshold search, future data in the selector or evaluation-role change.

## Two Contrasts, Two Different Questions

1. **Move the second edit, keeping the first.** The original same-query
   independent policy emits two edits at the first eligible pair query.
   The split policy emits the exact same first path there, then makes
   one independent edit at the first eligible later query. It does not
   know whether that query will occur. Compare utility only in the
   retrospective intersection where both actually make two edits;
   check recording risk on the entire roster for each policy.
2. **Coordinate across queries, keeping both clocks.** The early
   independent and early joint policies make the same first edit at the
   first eligible single query and make the second at the next eligible
   query. The independent choice ignores the first score. Joint choice
   adds the first edit's predicted gain and risk moments before applying
   the same penalties. Both second choices use the scale fixed at the
   first query and use no intervening future error, label or recording
   length. The preliminary mixed-scale registration is retained but
   is not the primary comparison.

The budgets are development probes, not new operating permissions.
The original same-query score reproduced in every location-seed cell.

| Policy | Edits | Risk passes | Undefined easy | Risk pass and beat CV and OLS4 / 606 | Short-roster joint passes / 105 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same-query independent | 744 | 64 | 432 | 48 | 2 |
| Split, same first edit | 729 | 84 | 426 | 70 | 2 |
| Early independent | 1,185 | 128 | 303 | 92 | 11 |
| Early joint | 1,185 | 125 | 301 | 89 | 11 |

On the **357 views with two edits in both timing arms**, moving the
second edit later yields **-0.267 pp [-0.621, -0.028]** direct
conservative gain. The retrospective subset has five supported locations,
and only three views are from the short roster. Fifteen other views
start but do not finish the split pair; 234 do not trigger either timing
arm. The larger full-roster pass count is not a matched utility gain or
a risk qualification.

Early joint changes 82 second choices among 579 completed pairs. Its
direct conservative gain over early independent is **+0.0033 pp
[-0.0036, +0.0139]** on the entire identical-clock roster. Each policy
has 27 one-edit views and 579 two-edit views, so this contrast is
matched without retrospective selection. The original full-recording
gate remains unmet. Confidence intervals average three seed views
before bootstrapping locations 3,000 times. The 606 views are 202
recordings, not 606 independent samples.

## Why Waiting Is Not the Main Bottleneck Here

Fix the actual first early edit on the 105 credit-starved views, then
exhaustively try each finite candidate for the second edit. The
next eligible query admits a qualifying witness in **16 views**.
Allowing any later eligible query gives **19**. The broader oracle
without that first-edit restriction previously found **30** cross-query
witnesses. All positive witnesses are rescored by the same evaluator.
The 3-view difference is one distinct recording: the choice of
first edit and the candidate bank matter more than another wait
threshold for this probe. The hindsight counts do not enter inference.

**Decision:** preserve the current operating policy. Do not promote
timing or two-action score aggregation as a contribution. Prioritize
the support and prediction quality of the first intervention over
additional variants of second-edit scheduling. These are exposed
development results; the 30 DroneCrowd groups remain sealed.

```sh
.venv-pytorch/bin/python scripts/run_m3w_cross_query_edit.py
```

Native arm64 Torch, four compute threads, zero data workers. The command
writes per-location receipts and heartbeat, verifies the 21 model
checkpoints, and skips already scored locations on replay. No training,
CREATE submission, deployment, metric/seconds claim, Stage5C or SMC.
Twenty-eight targeted tests passed; a replay produced byte-identical
scores from the verified site receipts.

[Registration](registration.json) | [Results and location values](analysis.json) |
[Preliminary mixed-scale result](../cross_query_edit_v1/analysis.json).
