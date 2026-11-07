# Predicting Missing Labels Does Not Improve Matched Neural Replacement

Each ranked policy makes 9,675 genuine changes to the same motion
recipients. It uses exactly 6,066 neural replacements, with the
same neural count in every query as the frozen CV-difference arm.
The motion-only anchor has the same total switch count but uses
no neural replacements. No count or threshold was chosen on
scored outcomes.

| Compared with CV difference | Available utility, pp | Conservative lower utility, pp | Recording passes / 102 |
|---|---:|---:|---:|
| Direct target, matched count | -0.039 [-0.217, +0.098] | -0.477 [-1.267, +0.026] | 0 |
| Minimum disagreement | -0.122 [-0.254, +0.021] | -0.469 [-1.102, -0.063] | 0 |
| Factorized, donor prevalence | +0.054 [-0.081, +0.178] | -0.247 [-0.731, +0.104] | 0 |
| Factorized, fitted probability | +0.043 [-0.080, +0.183] | -0.285 [-0.812, +0.099] | 0 |

The primary fitted-minus-constant comparison gives conservative
utility **-0.0628 pp** [-0.1513, +0.0045]. Fitting past-only
missingness changes mean Brier from 0.04418 to 0.03802 (paired
difference -0.00616 [-0.01858, +0.00154]). The direction of the
point estimate does not establish reliable prediction improvement,
and it does not produce decision utility. Observed relative-pair
contrast is -0.0034 pixels [-0.0547, +0.0603]; the full-grid lower
pair contrast is -0.336 pixels [-0.679, -0.087].

Both factorized arms reduce all-risk violations from 100 to 95
views versus CV difference, yet pass no recording under the full
rule. All rerankers still have 15 undefined selected-easy views.
These are repeated views of 34 exposed recordings. Intervals are
3,000 paired source-group draws after seed averaging.

**The recipient restriction is the larger obstacle.** The separate
[label-aware component audit](../native_fixed_recipient_risk_bound_v1/README.md)
rules out at least 99/102 views with fixed replacement counts,
and 93/102 even with free neural counts at the same recipients.
These findings do not justify changing the 2% rule. They prioritize
support-aware recipient retrieval over another path ranker on
this fixed set. No deployment or independent claim is made.

~~~sh
.venv-pytorch/bin/python scripts/run_m3w_visibility_increment.py fit
.venv-pytorch/bin/python scripts/run_m3w_visibility_increment.py freeze
.venv-pytorch/bin/python scripts/run_m3w_visibility_increment.py score
.venv-pytorch/bin/python scripts/run_m3w_visibility_increment.py replay
~~~

The 42 nested source/seed fits ran locally in 80.6 seconds.
The exact pre-score freeze took 3.6 seconds, scoring and full
re-score 22 seconds each. The experiment and its component audit
passed 10 targeted tests together. The released-label mask is
not physical visibility. All coordinates are image pixels,
with 8 observed and 12 predicted annotation steps.

[Protocol](protocol.md) | [Metrics](analysis.json) |
[Fit](fit_execution.json) | [Freeze](freeze_execution.json) |
[Replay](replay_execution.json)
