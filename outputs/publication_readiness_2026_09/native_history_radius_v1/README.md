# Near-CV Replacements Can Add Support Without Adding Useful Gain

I kept the 2,194 screened interventions and changed only their
forecast paths. The cap is estimated from other sources, using either
pooled motion scale or a past-only CV backtest inside the eight
observations. The third arm applies the history cap only if it permits
at least 0.25 pixels of displacement. That is a predeclared
sensitivity, not an estimate of annotation resolution.

| Same agents and switch count | Original risk passes / 102 | All-risk violations | Undefined easy risk |
| --- | ---: | ---: | ---: |
| Original | 4 | 40 | 62 |
| Pooled cap | 40 | 0 | 62 |
| History-conditioned cap | 40 | 0 | 62 |
| History cap with material displacement requirement | 4 | 40 | 62 |

The pooled and history caps change all 2,194 paths. Both fitted
reference estimates lie below the observed reference on all 1,188
selected easy instances in this exposed evaluation; that is not a
coverage guarantee on a new source. The material operating arm
changes none of the selected paths. The literal history cap reduces
conservative utility by **0.04553 pp** [-0.09475, -0.00917] versus
the original. It adds only +0.000096 pp over CV and remains
-9.82457 pp [-21.99826, -0.55518] versus OLS4. Intervals are
3,000 paired source draws after seed averaging.

| Hindsight bank | Risk-feasible / 102 | Verified positive OLS4 witnesses | Unresolved OLS4 optimizations |
| --- | ---: | ---: | ---: |
| Original full bank | 99 | 51 | 0 |
| Add literal nonzero clipped paths | 102 | 51 | 3 |
| Add only clipped paths at least 0.25 pixels from CV | 99 | 51 | 3 |

The original three failures were one recording repeated at three
seeds. Tiny nonidentity options make that recording feasible,
but the material candidates do not. The exact easy-reference
criterion verifies feasibility even when OLS4 optimization times
out. No new positive OLS4 witness is verified; an unresolved
optimization is not proof that no such witness exists.

**Decision:** Keep the frozen policy. Higher pass counts after nearly
reverting to CV are not evidence of useful prediction improvement.
This study does not turn abstention into a pass or change the 2% rule.
It uses 34 exposed recordings in 14 sources and three frozen seeds,
8 observed and 12 forecast annotation steps in pixels. No new
external access, deployment, Stage5C or SMC.

[Protocol](protocol.md) | [Metrics](analysis.json) |
[Freeze](freeze_execution.json) | [Replay](replay_execution.json)

```sh
.venv-pytorch/bin/python scripts/run_m3w_history_radius.py freeze
.venv-pytorch/bin/python scripts/run_m3w_history_radius.py score
.venv-pytorch/bin/python scripts/run_m3w_history_radius.py replay
```
