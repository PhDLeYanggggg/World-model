# Reopening Budget Vetoes Recovers Utility But Not Safety

The previous audit found easy-supported local choices that were blocked
by predicted risk. I refitted the all/easy positive-harm excess residuals
on donors, comparing all legal actions with training only on predicted
vetoes. The proposal model excludes both the donor and evaluation
sources. Both corrections apply only to the original single-action
veto region, preserving the original zero-credit count witness and
the exact per-query counts (6,184 switches in total). The utility objective and final
observed rules are unchanged.

| Policy | Lower Utility vs CV (pp) | All Risk Above 2% | Easy Risk Above 2% | Signed Easy Net Above 2% | Undefined Easy | Full Passes / 102 |
|---|---:|---:|---:|---:|---:|---:|
| Original | +0.10154 | 68 | 65 | 2 | 34 | 3 |
| All-action residual fit | +0.31239 | 67 | 60 | 8 | 39 | 3 |
| Veto-only residual fit | +0.31395 | 66 | 60 | 8 | 39 | 3 |

Veto-only vs original improves the **direct paired completion lower**
by +0.20044 pp [0.05235, 0.38133]. The all-action arm gives +0.19888
pp [0.05252, 0.37634]. The matched primary comparing veto-only
to all-action is only +0.00156 pp [-0.00110, +0.00496], so the
observed utility benefit is not specific to veto-only training.
Intervals are 3,000 paired draws of 14 sources after averaging
three actual, previously trained forecast banks. The original
2% risk and support checks still fail.

The targeted arm reopens 1,086,575 repeated row-action occurrences;
643,579 satisfy both local observed budgets at zero credit. Only
3,740 reopened actions are used in the final choices, of which
1,087 do not satisfy both local budgets at zero credit. A failed
local check is not itself a recording-level risk violation. The
reopening provides a locally admissible easy option in an active
query for three of the 33 previously limited views, but only one
ends up with defined selected easy reference. Worst signed easy net
degradation rises from 2.86% to 4.21%.

## What Drives The Tradeoff?

Hard known utility improves +0.25149 pp [0.05662, 0.47217] over
the original. On known query-weighted errors the reopened rows
account for +244.09 image-error mass and other rows for -2.79.
As posthoc bookkeeping, restoring original paths **only** on
newly selected reopened rows changes eight signed easy-net
violations to one. This hybrid was not count-matched or evaluated
as a policy and is not a deployment proposal. These comparisons
locate the utility and easy-harm tradeoff in the same part of the
decision, rather than showing that easy harm can be removed for free.

**Decision.** The cost refit is a meaningful development utility result,
not a safe policy. Reopening more candidates is insufficient: the
next repair should discriminate useful from harmful releases under
the original rule, with tuning confined to fitting sources and explicit
coverage reporting. Do not deploy this model or lower the 2% bar.

Exposure: Moving A development only, 8 observed and 12 future
annotation steps in image coordinates. No new neural forecaster,
external outcomes, physical-safety claim, Stage5C, or SMC.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_veto_risk_refit.py fit
.venv-pytorch/bin/python scripts/run_m3w_veto_risk_refit.py score
.venv-pytorch/bin/python scripts/run_m3w_veto_risk_refit.py replay
.venv-pytorch/bin/python scripts/audit_m3w_veto_refit_tradeoff.py
```

Registered at `3fc11917`. Eighty-four new source-excluded residual
fits complete in 58.45 s on native arm64 with per-source checkpoints;
decisions freeze before scoring. The original control and scalar replay
are exact. [Protocol](protocol.md) | [Replay](replay_execution.json) |
[Attribution](tradeoff_attribution.json)
