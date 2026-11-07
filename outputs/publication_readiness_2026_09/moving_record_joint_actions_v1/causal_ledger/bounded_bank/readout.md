# Smaller Forecast Changes Help, But Learning Their Allocation Still Fails

I added half- and quarter-displacement versions of each of eight frozen
alternatives to constant velocity. The original eight paths and full-strength
cost heads stay unchanged. I fitted 84 new source-excluded actionwise cost
heads: two amplitudes, fourteen held sources, three verified forecast-bank
seeds. The new heads predict the original gain and harm quantities for
each actual blended path. This is not residual training or another fit of
the trajectory backbones.

| Policy | Switches | Completion Lower vs CV (pp) | All Risk Upper Above 2% | Easy Risk Upper Above 2% | Undefined Easy | Original Passes / 102 |
|---|---:|---:|---:|---:|---:|---:|
| Full bank, fixed counts | 6,184 | +0.10154 | 68 | 65 | 34 | 3 |
| Same choices, half displacement | 6,184 | +0.07860 | 51 | 58 | 34 | 10 |
| Same choices, quarter displacement | 6,184 | +0.04531 | 19 | 39 | 34 | 29 |
| Expanded bank, fixed counts | 6,184 | +0.11837 | 70 | 66 | 36 | 0 |
| Full bank, free counts | 6,876 | +0.10516 | 70 | 67 | 28 | 7 |
| Expanded bank, free counts | 8,754 | +0.13802 | 76 | 74 | 19 | 7 |

The fixed-count comparison matches the number of interventions at **every**
query. The primary paired lower gain is +0.01573 pp [0.00396, 0.02800]
against the full bank (known gain +0.01706 pp [0.00468, 0.02988]).
It also beats the fixed-half and fixed-quarter controls on utility by
+0.03928 pp [0.01454, 0.06469] and +0.07282 pp [0.03412, 0.11276].
Intervals use 3,000 paired source draws after averaging three actual
forecast-bank seeds. They do not treat overlapping windows as independent.
The free-count contrast is +0.03018 pp [-0.00214, 0.05973]; it also changes
how often the model switches.

**Safety and utility separate.** At fixed counts the expanded bank chooses
4,436 full, 1,464 half and 284 quarter actions. Despite the added low-amplitude
options, 70 views exceed all-case completion risk and 66 exceed easy risk.
The simple quarter-displacement control improves both, and reduces signed
easy degradation above 2% from two views to none. Its selected easy support
does not change: 34 views stay undefined. The fixed-quarter direct
completion-lower difference against the full bank is -0.06880 pp
[-0.10715, -0.03489]; report the tradeoff, not just the safer side.
The study does not show that the learner can tell which interventions to
attenuate, or isolate the contribution of each choice to the risk increase.

For identical selected agents and positive amplitudes, convexity bounds
each observed positive ADE harm and missing-target radius by its amplitude
times the original quantity. This explains why uniform attenuation is
a useful control. It cannot certify unsupported views, substitute for
the 2% test, or transfer to a new allocation with different selected rows.
No method passes the original full-roster requirement. I do not deploy it.

The study addresses relative-to-baseline intervention decisions, not
classification deferral ([Mozannar and Sontag, ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html))
or joint forecast accuracy alone ([Weng et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.html)).
The allocation objective does not optimize collisions. This is an
exposed-A development result in image coordinates, eight observed
and twelve forecast annotation steps. No independent test was opened.
No metric, seconds-level, 3D, foundation, Stage5C or SMC claim follows.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_bounded_path_bank.py fit
.venv-pytorch/bin/python scripts/run_m3w_bounded_path_bank.py score
.venv-pytorch/bin/python scripts/run_m3w_bounded_path_bank.py replay
```

Design `384f0552` was committed before fitting. The native arm64 run froze
all three banks in 21.39 seconds with per-seed resume. Score and replay
both reproduce the original full-bank metrics and hash of the complete
readout. Twenty-nine scoped tests pass. Per-view analysis and caches stay
local. [Protocol](protocol.md) | [Score replay receipt](replay_execution.json)
