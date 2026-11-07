# Continuous Benefit And Harm Activation

**Decision: do not promote.** Estimating magnitudes does not reliably
improve matched utility over the previous gain score. The tested
risk penalty reduces utility without increasing recording passes.

The [registered protocol](protocol.md) reuses the exact candidate
choices and gate of the activation study, with ten fitting sources,
three calibration sources and one scored source per fold. Forty-two
fresh folds fit 210 boosted regressors. The forecasts are frozen
re-use, not fresh training. The 102 recording-seed views are repeats
of 34 exposed recordings. Paired intervals use 3,000 draws over 14
source groups after seed averaging.

| Rule | Agent switches | Local hits | Original passes / 102 | Undefined easy risk | All-risk upper >2% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous gain, matched | 430 | 59 | 1 | 78 | 29 |
| Probability, matched | 430 | 74 | 5 | 76 | 27 |
| Continuous value, matched | 430 | 68 | 3 | 79 | 25 |
| Continuous value-risk, matched | 430 | 70 | 3 | 79 | 26 |
| Previous gain, current query | 310 | 20 | 0 | 90 | 15 |
| Continuous value, current query | 470 | 52 | 2 | 84 | 19 |
| Continuous value-risk, current query | 516 | 64 | 2 | 83 | 22 |

The primary comparison uses entire-recording score ranks only to
match counts. Risk-adjusted value minus previous gain is **+0.00105 pp**
[-0.00541, +0.00788] in conservative CV-reference utility. Value alone
is **+0.00251 pp** [-0.00445, +0.01054] versus previous gain. The tested
penalty costs **0.00146 pp** [0.00022, 0.00305] versus value alone.
Observed relative-path ADE reduction versus previous gain is
+0.00096 pixels [-0.00159, +0.00486]; complete-grid bounds span zero
[-0.00306, +0.00434] pixels. This does not establish a pair benefit.

The current-query risk-adjusted arm improves the conservative lower
metric by +0.00286 pp [+0.00011, +0.00635] versus previous current-query
gain, **but switches 516 times versus 310**. It has no clear conservative
gain versus CV and no demonstrated win over OLS4. Its lower OLS4
contrast is -9.814 pp [-21.986, -0.541]; this is a lower bound, not
proof of a negative complete-label effect.

On the 483 eligible selected seed-queries, normalized absolute errors
are 0.461 for conservative gain, 0.226 and 0.175 for all/easy harm, and
1.053 and 0.328 for all/easy reference. These are error scales, not proof
that the model is calibrated. The measured intervention counts and
recording-level failures remain visible. As in prior
[multi-expert deferral work](https://proceedings.mlr.press/v235/mao24d.html),
continuous cost modeling is not a standalone novelty claim.

The next useful question is whether the available forecast geometry
admits a materially better accuracy/risk tradeoff, rather than whether
another score can swap a few already sparse interventions. Measure that
capacity on the same recordings first, then change the candidate
forecast or features only where the diagnosis supports it.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_native_value.py fit
.venv-pytorch/bin/python scripts/run_m3w_native_value.py score
.venv-pytorch/bin/python scripts/run_m3w_native_value.py replay
```

[Analysis](analysis.json), [fit](fit_execution.json), [score](score_execution.json),
[replay](replay_execution.json). Original 2% and reference-support rules,
units (image pixels and annotation steps), deployment and reserved data
roles are unchanged. Stage5C and SMC remain off.
