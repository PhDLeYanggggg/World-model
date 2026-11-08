# Does Learning Recording Qualification Improve the First Edit?

The previous [cross-window comparison](../cross_query_edit_v2/README.md)
found little value in changing the second edit. Here the edit budget is
exactly one and the clock is the first eligible query. All edited arms
use the same candidate bank, past motion inputs, eight observed and
twelve requested steps, OpenDD source roles, and unchanged CV-relative
2% recording rule. CV with no edit remains a separate comparator;
its selected-risk denominator is undefined, not a failure of its
forecast accuracy.

## Targets and Comparators

For each training candidate, compute the complete one-edit
counterfactual against CV and OLS4. The edit must supply positive easy
reference mass, pass all original risk constraints, and strictly
improve on both references under the missing-step debit. The OLS4
offset on the rest of the recording is part of this target, not an
input. Each positive risk or utility witness used in the ceiling
is independently checked with the original evaluator.

One new shared head predicts candidate gain and risk moments, the
qualification event directly, and the factors
`P(easy) P(safe | easy) P(qualifies | safe)`. Conditional losses are
applied only where the condition holds. Three readouts of this head
give a matched comparison; the older frozen moment head is a separate
control. These scores are not calibrated probabilities. Fitting uses
only the six non-held development locations; training targets use
easy cuts excluding both the held and donor locations. Sampling
balances locations, then recordings, then candidates. No held-site
threshold selection or candidate-bank change.

| First-query policy | Edits | Risk passes | Undefined easy | Joint passes / 606 | Short-roster passes / 105 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CV without intervention | 0 | 0 | 606 | 0 | 0 |
| Frozen moments | 606 | 107 | 414 | 76 | 5 |
| Refit moments from shared head | 606 | 63 | 443 | 43 | 5 |
| Direct qualification | 606 | 92 | 434 | 67 | 3 |
| Factorized qualification | 606 | 101 | 426 | 74 | 6 |

Compared with the refit moment readout, direct and factorized
qualification gain **-0.087 pp [-0.676, +0.357]** and
**-0.079 pp [-0.665, +0.368]**, respectively, in paired conservative
utility. Thus the event heads recover risk-and-utility passes over
their shared-head control without establishing a reliable forecast
gain. Neither exceeds the original frozen head's pass count.
Intervals use 3,000 paired location bootstrap draws after averaging
three seeds. The 606 recording-seed views represent 202 recordings,
not 606 independent samples.

## When a Useful First Edit Exists

| Hindsight option space, same one-edit budget | Full views / 606 | Credit-starved views / 105 |
| --- | ---: | ---: |
| Qualifying candidate at the first eligible query | 120 | 15 |
| Qualifying candidate at any eligible query | 441 | 30 |

At the first query, 306 views have an editable easy reference and
186 have an edit that passes risk. Only 120 can also beat both
baselines. The exact comparison shows that forcing the first eligible
query discards a substantial portion of the one-edit capacity on
this roster. It does not prove a past-only controller can find the
later windows, or that waiting is costless. The effect is smaller
on the original short-roster subset.

**Decision:** no first-query readout qualifies. The next informative
comparison is an online one-edit stopping rule, selected only by
out-of-fold donor predictions, against fixed-delay and immediate
controls. The key question is whether later qualification can be
identified before editing, not whether a stronger oracle can move
the same edit after seeing its outcome.

```sh
.venv-pytorch/bin/python scripts/run_m3w_first_edit_target.py
.venv-pytorch/bin/python scripts/run_m3w_first_edit_target.py --replay
```

21 fresh native arm64 Torch fits, 10,500 optimizer updates, four
compute threads and no data workers. Checkpoint and heartbeat every
100 updates. The zero-update replay gave identical aggregate scores.
Twenty-five targeted tests passed. No deployment, CREATE submission,
metric/seconds claim, Stage5C or SMC. The 30 DroneCrowd confirmation
groups remain sealed.

[Registration](registration.json) | [Scores](analysis.json) |
[Checkpoint receipts](head_freeze.json).
