# Direct Feasibility Ranking On The Fixed Slate

**Decision: no promotion.** Predicting the local joint event directly
does not improve the registered top-one endpoint or repair recording
risk. The original 2% rule and undefined-reference requirements remain.

The [registered comparison](protocol.md) uses 38,682 options on 4,839
seed-queries from 34 previously exposed recordings and 14 source groups.
Forty-two fresh boosted rankers use three actual forecaster seeds.
The forecast bank is verified reuse. To construct each donor slate,
the additive proposer excludes both the scored source and the donor
source. The ranker trains on the other 13 sources. The 102 views are
not independent recordings. Intervals use 3,000 paired source draws
after seed averaging.

| Ranking and operating mode | Switches | Locally feasible sets | Views retaining a feasible set | Original recording passes |
| --- | ---: | ---: | ---: | ---: |
| Support rank, all queries | 9,675 | 755 | 67 | 1/102 |
| Feasibility rank, all queries | 9,675 | 759 | 65 | 0/102 |
| Support rank, old gate | 222 | 40 | 18 | 5/102 |
| Feasibility rank, same active queries and gate | 222 | 38 | 18 | 2/102 |

The registered all-query top-one feasibility effect is **-0.169 pp**
[-2.058, +1.932] after source balancing; the unweighted count increases
by four. At all-query matched counts, conservative utility drops by
**0.252 pp** [0.001, 0.638], in CV-reference units. Observed relative-path
ADE improves by **0.076 pixels** [0.006, 0.164], but missing-step
completion bounds [-1.113, +1.183] do not support a full-grid advantage.

At the same 222 switches and with the same predicted gate, 47 queries
change candidates. Conservative utility improves only **+0.00118 pp**
[+0.00004, +0.00309] relative to old gated support, an exploratory
secondary contrast. There is no reliable gain over CV, no positive
conservative comparison to OLS4, and no clear relative-geometry gain
in this operating mode. Both arms have 75/102 views with undefined easy
risk and 22/102 exceeding the all-risk cap.

## Where Ranking Runs Out Of Room

The [checkpoint audit](checkpoint_audit.json) reloads the 42 models and
regenerates all held-query scores and choices exactly. It then checks
the labeled options without modifying decisions. Only **49/111** old
active queries contain an actually feasible gate-eligible choice; the
new ranker finds 38. Conversely **84 inactive queries** have such a
choice, and 1,564 queries have a feasible choice before the predicted
gate. These are label-aware opportunities, not selectable known actions.
The fixed mask cannot be repaired by ranking alone. A next experiment
should compare learned activation on donor-calibrated queries against
the old schedule at matched intervention count, without replacing
the final recording check with a local event.

The method estimates a local event; it is not a risk guarantee.
[Selection-conditioned risk control](https://proceedings.mlr.press/v306/wang26a.html)
already uses linear expectation constraints for a marginal conditional
error estimand, not these recording-wise continuous-reference checks.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_native_feasible_slate.py fit
.venv-pytorch/bin/python scripts/run_m3w_native_feasible_slate.py score
.venv-pytorch/bin/python scripts/run_m3w_native_feasible_slate.py replay
.venv-pytorch/bin/python scripts/audit_m3w_native_feasible_slate_checkpoint.py
```

[Analysis](analysis.json), [fit](fit_execution.json), [score](score_execution.json)
and [replay](replay_execution.json). Checkpoints and decision arrays stay
local. This is development evidence in dataset pixels and annotation
steps, not external confirmation, deployment, calibrated metric/time
evidence, Stage5C execution, or SMC.
