# Native Query Activation At Matched Counts

**Decision: no deployment change.** The probability of a locally feasible
switch is not an adequate activation score by itself. It finds more local
hits but reduces matched conservative utility and does not repair the
original 2% recording rule.

The [registered comparison](protocol.md) refits the gate, reference-support
head and feasibility classifier on ten sources per fold, reserves three
for threshold calibration and scores one held source. The forecast bank
and reference cuts are reused with verified provenance. There are 42
fresh folds across three forecaster seeds and 14 exposed source groups.
The 102 recording-seed views are repeats of 34 recordings. Intervals use
3,000 paired source draws after averaging seeds. No independent data are used.

| Policy | Agent switches | Locally feasible queries | Original rule passes / 102 | Undefined easy risk | All-risk upper >2% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prior frozen support (context only) | 222 | -- | 5 | 75 | 22 |
| Refit support control | 430 | 68 | 2 | 76 | 33 |
| Matched offline gain | 430 | 59 | 1 | 78 | 29 |
| Matched offline probability | 430 | 74 | 5 | 76 | 27 |
| Current-query gain | 310 | 20 | 0 | 90 | 15 |
| Current-query probability | 376 | 79 | 2 | 85 | 15 |

**Equal-count diagnostic.** The two offline arms choose the same action
whenever they activate the same query. They each activate 215 queries,
with exact matching within recording and agent-count bucket. They rank
scores over the full recording, which prevents using them as online
policies. Probability minus gain on the registered paired lower utility
is **-0.00388 pp [-0.00910, -0.00003]**, with a common CV denominator.
Observed relative-path ADE worsens by **0.00211 pixels**
[0.00022, 0.00457]. The local hit count increases, but the source-balanced
increase is +0.130 pp [0.000, +0.304].

**Current-query thresholds.** The two scores each trigger 375 donor
query occurrences across the 42 calibration folds. After transport
to the scored sources, the counts differ (155 versus 188 queries).
Probability gains +0.00551 pp [0.00000, +0.01582] versus CV, and
-0.00221 pp [-0.01576, +0.00808] versus current-query gain. Its lower
utility versus OLS4 is -9.819 pp [-21.998, -0.555]; this does not
establish superiority or prove a negative complete-data effect.
Both current-query rules leave many recordings unevaluable or over
the original risk cap. Matching donor activation frequency is not a
risk guarantee; [selection-conditioned risk control](https://proceedings.mlr.press/v306/wang26a.html)
targets a different marginal estimand with its own assumptions.

The binary local event ignores the magnitude of gains and harms. A
follow-up should estimate continuous completion-lower gain and the
harm/reference margins on the chosen sets, then compare value-aware
activation without changing the original rule. The current data do
not identify a reliable new policy or justify opening confirmation.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_native_activation.py fit
.venv-pytorch/bin/python scripts/run_m3w_native_activation.py score
.venv-pytorch/bin/python scripts/run_m3w_native_activation.py replay
```

[Chinese instructions](reproduction_zh.md), [analysis](analysis.json),
[folds](freeze.json), [fit](fit_execution.json), [score](score_execution.json)
and [replay](replay_execution.json). No changes to data roles, deployment,
Stage5C or SMC. Units remain image pixels and annotation steps.
