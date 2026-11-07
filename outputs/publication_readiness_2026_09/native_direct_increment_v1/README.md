# A Direct Neural-Increment Target Does Not Beat The CV-Difference Control

The new ridge heads predict the gain of a neural path over the
already selected motion path. All three unscreened policies
have the same 9,675 genuine switches on exactly the same agents
and queries. The original 2% rule still uses CV as its risk
reference. No neural trajectory generator was retrained.

| Policy | Total switches | Neural replacements | Original rule passes / 102 | All-risk >2% | Worst signed easy degradation |
|---|---:|---:|---:|---:|---:|
| Retain causal anchor | 9,675 | 0 | 0 | 102 | 81.269% |
| CV-difference target | 9,675 | 6,066 | 0 | 100 | 135.000% |
| Direct increment target | 9,675 | 8,090 | 1 | 96 | 265.071% |
| Retain causal anchor, screened | 78 | 0 | 0 | 18 | 0.046% |
| CV difference, screened | 92 | 9 | 0 | 19 | 0.046% |
| Direct increment, screened | 8 | 8 | 1 | 3 | 0.000% |

| Paired comparison | Available-label utility, pp | Completion lower utility, pp |
|---|---:|---:|
| Direct vs. CV difference | -0.055 [-0.157, +0.037] | -3.303 [-9.403, -0.044] |
| Direct vs. retain anchor | +0.151 [-0.034, +0.357] | -2.921 [-8.758, +0.124] |
| CV difference vs. retain anchor | +0.206 [+0.007, +0.426] | -0.291 [-1.068, +0.210] |

The **3.247 pp** gap between the available and lower point contrasts
against CV difference is the extra disagreement allowance on wholly
unobserved rows. It is not measured error on those rows. The new
method also worsens observed relative-pair ADE against the anchor by
0.336 pixels, with a signed reduction of -0.336 [-0.737, -0.088].
Its separately screened policy leaves 98 views without an estimable
selected easy ratio, so one pass does not establish a usable policy.

Pooled available ADE / endpoint FDE (pixels) are 32.739 / 70.004
for the anchor, 32.691 / 69.863 for the CV difference, and
32.673 / 69.834 for the direct target. These pooled statistics use
368,148 known rows and 278,448 known endpoints across repeated
seeds. They are descriptive, not the source-balanced paired
estimand or a substitute for the risk checks.

**Decision:** Do not promote the direct critic. Changing the
regression reference alone increased the number of neural choices
without preserving either conservative gain or easy outcomes.
The discrepancy between the observed and conservative results
prioritizes modeling the disagreement on potentially missing futures,
and testing whether that information helps at the same number of
neural replacements, rather than adding another forecast family.

## Reproduction And Scope

~~~sh
.venv-pytorch/bin/python scripts/run_m3w_direct_neural_increment.py fit
.venv-pytorch/bin/python scripts/run_m3w_direct_neural_increment.py freeze
.venv-pytorch/bin/python scripts/run_m3w_direct_neural_increment.py score
.venv-pytorch/bin/python scripts/run_m3w_direct_neural_increment.py replay
~~~

The fit produced 42 excluded-source ridge checkpoints (14 sources,
three forecaster seeds), using 12-source anchor fits on 13 donor
sources per held-source head. It rechecked the original causal
coefficients from source moments. The local fit, past-only freeze,
score, and exact re-score took about 80, 7, 18 and 18 seconds.
Eight targeted tests passed. The 102 risk views repeat 34 exposed
recordings. All confidence intervals use 3,000 paired draws of
whole sources after seed averaging. No external outcomes, no new
threshold selection or deployment, no Stage5C execution or SMC.
All forecast distances remain image pixels at 8/12 annotation steps.

[Protocol](protocol.md) | [Metrics](analysis.json) |
[Fit](fit_execution.json) | [Freeze](freeze_execution.json) |
[Replay](replay_execution.json)
