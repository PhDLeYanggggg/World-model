# Neighbor Context Does Not Improve Fixed-Shortlist Intervention

I tested whether current-query interactions help choose a better
action from the same eight candidates. The observations are eight
annotation steps and the forecasts are twelve annotation steps in
native image coordinates. This is the previously exposed Moving A
development cohort, not an independent test.

The first model adds six features from synchronized past observations:
nearest distance now and over the observation window, two measures of
recent closing, heading agreement, and local density. The second
adds six features measuring baseline proximity and the change when
one agent takes a candidate path while its neighbors stay on their
causal floors. Both models retain the same forecasts, shortlist,
loss, budget, donor calibration and direct interventions as the
unaugmented risk-aware selector. No ground-truth future enters the
feature calculation. The rollout quantities are relative image-space
proximity proxies, not collisions or physical safety.

| Calibrated policy | Switches | Complete passes / 102 | Signed easy loss >2% | Easy upper bound undefined |
|---|---:|---:|---:|---:|
| Direct only | 6,952 | 2 | 8 | 35 |
| Fixed shortlist without interaction inputs | 6,955 | 2 | 8 | 35 |
| Add observed neighbor motion | 6,955 | 2 | 8 | 35 |
| Add observed motion and rollout proximity | 6,959 | 2 | 9 | 34 |

| Added input | Calibrated paired difference [95% interval] | Matched paired difference [95% interval] |
|---|---:|---:|
| Observed neighbor motion | 0 [0, 0] | -0.00853 [-0.02540, +0.00176] |
| Observed motion and rollout proximity | +0.00256 [0, +0.00628] | -0.00695 [-0.02996, +0.00797] |

Differences are percentage points of completion-lower utility versus
the unaugmented selector, with 3,000 paired resamples of 14 sources
after averaging the three forecaster seeds within each source. The
102 views are 34 recordings repeated across those seeds. At exactly
6,998 switches on the same queries, all three selectors pass 5/102
views, but both interaction arms have 14 signed easy violations
versus 11 for the unaugmented selector. This diagnostic does not
certify the complete policy.

In the first two eligible queries of the 18 views where the original
direct policy never switched, the full candidate bank contains a
positive supported choice in 15/18 (verified previous result).
The unchanged shortlist retains one in 6/18; each tested selector
recovers 0/18. The fixed-size shortlist therefore still loses many
opportunities before the final model can act, and neither neighbor
summary solves the remaining ranking error. This does not imply that
interaction-aware trajectory prediction is ineffective: it only
rejects the registered intervention improvement for these features
and this selector. A physically plausible-looking rollout proxy is
not a substitute for observing lower baseline-relative forecast
error and adequate reference support.

## Relation To Prior Work

[Social LSTM](https://openaccess.thecvf.com/content_cvpr_2016/html/Alahi_Social_LSTM_Human_CVPR_2016_paper.html)
uses neighborhood pooling to forecast interacting trajectories.
[Interpretable Social Anchors](https://openaccess.thecvf.com/content/CVPR2021/html/Kothari_Interpretable_Social_Anchors_for_Human_Trajectory_Forecasting_in_Crowds_CVPR_2021_paper.html)
models interpretable interaction structure for forecasting. This
ablation does not reproduce either predictor or claim novelty from
using neighbor motion. It tests a separate question: does an
interaction descriptor improve a baseline-relative decision with
predictions and switch counts held fixed?

## Reproduction And Scope

Registration `74064187` precedes all 84 new selector fits. Existing
forecaster outputs and the exact unaugmented decisions are hash-checked
parents. Each source/seed cell has seven fitting sources, six
calibration sources, and one held source. Only light evidence is
tracked; the private input bank and fold checkpoints are required.
Run from the repository root:

```sh
.venv-pytorch/bin/python scripts/run_m3w_shortlist_interaction.py fit
.venv-pytorch/bin/python scripts/run_m3w_shortlist_interaction.py score
.venv-pytorch/bin/python scripts/run_m3w_shortlist_interaction.py replay
```

The fit resumes from completed source checkpoints. The scoring check
requires unchanged direct actions and exact control risk/recall
counts. Score replay reproduces the aggregate bytes. The original
2% and support requirements stand. Neither model is promoted. No
reserved external outcomes, Stage5C execution, SMC, deployment, or
submission-readiness claim.
