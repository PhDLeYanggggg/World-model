# Choosing Recipients By Predicted Risk Shortfall Does Not Improve Matched Utility

The gain, support-first, and risk-shortfall rules choose from
the same nominated path for each agent. All make 9,675 genuine
changes at identical counts per query. Only the selected recipients
differ. They use the same frozen source-excluded cost fits; there
is no new training or outcome-selected threshold.

| Policy | Switches | Passes / 102 | Undefined easy risk | All-risk >2% | Worst signed easy degradation |
|---|---:|---:|---:|---:|---:|
| Gain | 9,675 | 4 | 23 | 88 | 98.702% |
| Support-first | 9,675 | 0 | 12 | 95 | 98.702% |
| Risk shortfall | 9,675 | 2 | 16 | 96 | 37.112% |
| Gain, own screen | 228 | 3 | 77 | 21 | 0.139% |
| Support-first, own screen | 88 | 4 | 86 | 14 | 0.046% |
| Risk shortfall, own screen | 4,312 | 4 | 47 | 55 | 1.743% |

At matched counts, shortfall minus support-first gives
**-1.991 pp** [-3.195, -0.927] conservative utility. Its
available-label contrast is -0.315 pp [-0.719, +0.082].
Against gain ranking the conservative contrast is
-1.305 pp [-3.142, -0.166]. Shortfall reduces the worst
signed easy degradation relative to the other raw arms, but
does not improve the original rule passes over gain ranking.
Observed relative-pair gain versus support-first is +0.246
pixels [-0.008, +0.564]; the full-grid lower contrast is -2.746
pixels [-4.921, -1.177]. It does not establish a pair benefit.

The own-screen shortfall arm appears better in lower utility than
the own-screen support-first arm (+0.046 pp [0.020, 0.074]),
but it changes 4,312 paths versus 88 and passes the same four
views. All-risk violations grow from 14 to 55. This is not a
matched benefit and it is not a successful deployment setting.

Separate label-aware minima rule out three views at these same
exact counts, whether using the fixed nominations or all native
paths. The other views are not certified by the separate minima.
The [simultaneous feasibility follow-up](../native_exact_count_joint_capacity_v1/README.md)
tests those constraints together.

## Reproduce

~~~sh
.venv-pytorch/bin/python scripts/run_m3w_recipient_shortfall.py freeze
.venv-pytorch/bin/python scripts/run_m3w_recipient_shortfall.py score
.venv-pytorch/bin/python scripts/run_m3w_recipient_shortfall.py replay
~~~

The three seed choices froze locally in 8.5 seconds, and the
score and exact re-score took about 21 seconds each. The gain
control replays the existing mixed independent choices exactly.
The 102 views repeat 34 exposed recordings; intervals use
3,000 paired draws over 14 source groups after seed averaging.
This is an 8/12 annotation-step result in image pixels. No
external access, model promotion, Stage5C or SMC.

[Protocol](protocol.md) | [Metrics](analysis.json) |
[Freeze](freeze_execution.json) | [Replay](replay_execution.json)
