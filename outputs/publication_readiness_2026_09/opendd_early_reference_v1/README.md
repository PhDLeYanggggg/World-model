# Can We Release Reference Support Earlier?

[Gain deferral](../opendd_gain_wait_v1/README.md) most often expired
because support was missing or late. Here the gain query, recipient,
path and radius stay fixed. The learned head can use the original
support slot before or at gain. If original support fires first,
it takes priority. An early edit suppresses, rather than adds to,
any later original support edit. Evaluation uses native 8 observed
and 12 predicted steps.

## Two Training Objectives

1. **Local feasible reference.** Predict whether a support candidate
   contributes a positive easy reference with locally bounded harm.
   Compare incidence-only to incidence times predicted conditional
   normalized reference mass. Neither head moves the original policy
   at the fixed 0.5 incidence gate.
2. **Recording advantage.** Learn the consequence of using early
   support instead of the complete original policy. Targets include
   paired conservative utility and transitions in risk, reference
   availability and joint pass. The gate is the positive part of
   the 90th percentile of training-recording maximum scores, fixed
   without held-location scores or labels. This is a donor-derived
   action budget heuristic, not independent risk calibration.
   The utility-only readout omits the risk/reference transition
   target in its score, sharing the fitted trunk.

Upstream forecasts and nomination heads exclude both the donor and
scored locations for training features. Original gain and support
paths replay before any comparison. The easy control uses the
recording-advantage arm's exact support time, count and radius,
ranking current candidates by the old easy score without amplifying
any direction. It is a matched comparison, not an independent policy.

| Policy | Edits | Joint pass | Risk pass | Undefined | Old reference lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 943 | 138 | 172 | 295 | 0 |
| Local mass or incidence | 943 | 138 | 172 | 295 | 0 |
| Recording advantage | 984 | 131 | 164 | 305 | 19 |
| Utility-only readout | 966 | 132 | 164 | 304 | 9 |
| Easy rank, same time/radius/count | 984 | 131 | 164 | 305 | 19 |
| Outcome-aware early-support menu | 965 | 188 | 219 | 260 | 0 |

On 606 exposed recording-seed views, recording advantage adds 41
support edits and advances 45. It gains +0.00183 pp against the
original [95% CI +0.00007, +0.00419], but loses 19 already defined
references and restores only 9. All 19 losses necessarily come from
replacements: a new support cannot remove an old edited path when
there was no original support. The matched easy rank changes only
five recipients and gives a head-minus-control effect of +0.00004 pp
[0, +0.00011]. These are 3,000 location-paired resamples, after
averaging three seeds. The intervention has not passed the risk gate.

The exact witness search finds 50 more jointly passing views while
keeping existing defined references. Of these, 28 advance support
at the original count and 22 add a missing support edit. The latter
22 additions are concentrated in two of seven locations. These are
outcome-aware opportunity counts and do not establish online lift.
The 19 losses point to a more specific next test: train a separate
replacement-loss target, evaluate added and replaced support
separately, and ask whether a policy improves reference coverage
without increasing harm under the original scorer.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_early_reference.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_early_reference.py --replay
.venv-pytorch/bin/python scripts/run_m3w_opendd_early_advantage.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_early_advantage.py --replay
```

`fresh_run`: 42 local-head fits (21,000 updates, PID 77971, 19.0 s)
and 21 recording-advantage fits (10,500 updates, PID 78234, 14.7 s)
on arm64 Torch with four threads and no loader workers; checkpoints
and heartbeats every 100 steps. `cached_verified`: frozen upstream
roles and paths, both zero-update replays. `not_run`: independent
confirmation or new CREATE work. DroneCrowd remains 15 calibration
groups and 30 sealed confirmation groups. The 2% rule and undefined
risk handling are unchanged. No deployment, submission, Stage5C or SMC.
