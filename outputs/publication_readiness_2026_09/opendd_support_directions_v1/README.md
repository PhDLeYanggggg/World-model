# More Support Directions at the Same Radius

The [frozen proposal audit](../opendd_reference_loss_v1/README.md)
showed little room to improve by thinning. This study holds each
recording's original gain fixed and expands the early support bank
with OLS4 and an analytic direction selected on observed prefixes.
Each direction is eligible only if it can be scaled **down** to
the row's original neural support radius; duplicates are removed.
The original support trigger has priority. A supplemental edit can
fire before or at gain and only once.

Both heads learn conservative utility and joint/risk transitions
against the complete original policy, plus a separate reference-loss
logit. The advantage cutoff is set by the 90th percentile of
training-recording maximum scores and floored at zero. The guard
also requires a negative loss logit. No held scores or outcomes
set either threshold. This is a development budget rule, not
independent risk calibration. The neural-only head is retrained.
The direction-only control replays the expanded head's exact row,
query, radius and count using the native neural path.

| Arm | Edits | Joint pass | Risk pass | Undefined |
| --- | ---: | ---: | ---: | ---: |
| Original | 943 | 138 | 172 | 295 |
| Frozen early policy | 984 | 131 | 164 | 305 |
| Retrained neural-only | 971 | 131 | 164 | 301 |
| Expanded bank | 974 | 132 | 164 | 300 |
| Neural, exact row and radius match | 974 | 133 | 165 | 300 |
| Frozen trigger, neural oracle | 945 | 142 | 175 | 293 |
| Frozen trigger, expanded oracle | 948 | 147 | 180 | 290 |
| Early window, neural oracle | 965 | 188 | 219 | 260 |
| Early window, expanded oracle | 975 | 203 | 233 | 249 |

The 606 views are recording-seed views from seven exposed locations.
The learned expanded policy used 18 alternative directions and lost
15 defined references. Its paired conservative utility difference
versus exact matched neural is +0.00123 pp [-0.00114, +0.00389],
and versus original is +0.00110 pp [-0.00103, +0.00459].
Intervals use 3,000 location-paired resamples after averaging three
seeds. The expanded arm does not establish a utility or risk win.
The outcome-aware rows keep the original where it already passes,
may add support, preserve defined references, and are not
equal-budget causal results.

## Why the Learned Head Missed the Feasible Actions

The expanded early window contains 65 feasible views where the
original does not jointly pass. The trained policy recovers one:

| Attribution among the 65 | Views |
| --- | ---: |
| Never releases; feasible scores below the advantage cutoff | 55 |
| Never releases solely because of the reference-loss gate | 0 |
| Releases at a query with no feasible action | 2 |
| Feasible action at that query, but at another recipient | 0 |
| Feasible direction at the chosen recipient, wrong direction selected | 7 |
| Joint pass recovered | 1 |

The trained policy also destroys seven original joint passes.
The audit replays the frozen head and reconciles the counts with
the original scoring rule. The next experiment should train on
the explicit recording-level pass transition instead of relying
on a composite expected-advantage score to admit actions. Compare
at matched intervention counts and report both new passes and
destroyed original passes. Adding more directions alone does not
address the 55 absent releases.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_support_directions.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_support_directions.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_support_directions.py
```

`fresh_run`: 42 fits, 21,000 updates, four native arm64 Torch
threads, zero loader workers, PID 80088, 26.4 seconds. Checkpoints
and heartbeats every 100 steps. `cached_verified`: upstream donor
lineage, zero-update replay, frozen-head miss attribution.
`not_run`: new CREATE training and independent confirmation.
Original recording-level 2% checks and undefined handling remain
unchanged. DroneCrowd confirmation stays sealed. No deployment,
submission, Stage5C or SMC.
