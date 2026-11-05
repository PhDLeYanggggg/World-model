# Fixed Warm-Error Priority Training

I tested a single objective change: give the warm model's observed false-safe
fitting errors weight 10, retaining the full original MSE anchor. Both arms
start from identical kinematic neural weights and receive 2,000 updates. The
fixed error mask is supervision only, never an inference feature. Unknown
outcomes keep their original bounds and are not reclassified as observations.
This is standard hard-example replay, not a novel dynamics architecture.

## Execution

fresh_run: six neural risk fits, 12,000 primary updates, another 2,000-update
full refit, a 400-update resume pilot, inference and exact readout replay.
Training: 515.49 seconds, peak RSS 7.506 GB,
native arm64 CPU, four compute threads, one interop thread, zero data workers.
All six fits freeze before evaluation.
cached_verified: source inputs, warm neural models, forecasts, router,
normalizers, old rules and all parent predictions/choices/metrics.
not_run: new trajectory dynamics, independent H, new producer seeds,
full expanded study or deployment. One-seed exposed SOURCE is development.
The initial zero-target synthetic gradient fixture failure is retained in
[the preflight amendment](preflight_amendment.md); it was fixed before training.

## Full Fitting Loss

Exact empirical-query weighted full-data losses, not minibatch scores.
The two objectives differ; compare both anchor and priority components, not
unlike total losses. Neither loss nor weighted score is a calibrated risk.

| Context | Arm | Full anchor MSE | Fixed priority MSE | Combined weight10 MSE |
|---|---|---:|---:|---:|
| eqmotion_fold1_seed17_critic0 | warm | 0.020248872 | 0.000231157 | 0.022329285 |
| eqmotion_fold1_seed17_critic0 | continued_mse | 0.011936338 | 0.000187208 | 0.013621212 |
| eqmotion_fold1_seed17_critic0 | false_safe_priority | 0.012278170 | 0.000020204 | 0.012460010 |
| eqmotion_fold0_seed17_critic1 | warm | 0.018960970 | 0.000401391 | 0.022573488 |
| eqmotion_fold0_seed17_critic1 | continued_mse | 0.011636327 | 0.000296616 | 0.014305875 |
| eqmotion_fold0_seed17_critic1 | false_safe_priority | 0.014343568 | 0.000117985 | 0.015405430 |
| eqmotion_fold0_seed17_critic2 | warm | 0.893710541 | 0.010539774 | 0.988568507 |
| eqmotion_fold0_seed17_critic2 | continued_mse | 0.680023855 | 0.009331559 | 0.764007882 |
| eqmotion_fold0_seed17_critic2 | false_safe_priority | 0.683036298 | 0.003923238 | 0.718345438 |

## Held SOURCE Results

| Policy | Known floor gain % | Hard gain % | Switch % | Full risk/support | All/easy violations | Undefined selected easy |
|---|---:|---:|---:|---:|---|---:|
| mean_veto | 4.091202 | 5.035472 | 21.307681 | 0/17 | 13/15 | 0 |
| readout_veto | 4.323320 | 5.354744 | 22.011498 | 0/17 | 13/15 | 0 |
| kinematic_veto | 3.941761 | 4.858158 | 17.315501 | 0/17 | 14/14 | 2 |
| continued_mse_veto | 3.600559 | 4.625653 | 15.014426 | 0/17 | 14/13 | 2 |
| false_safe_priority_veto | 3.032291 | 3.850492 | 13.585568 | 0/17 | 14/13 | 2 |

| Scope | Contrast | Known pp | Completion lower pp | Completion upper pp |
|---|---|---:|---:|---:|
| fit | priority_minus_affine | 0.069998 | -0.000889 | 0.140885 |
| fit | priority_minus_continued | -0.499210 | -0.529482 | -0.468939 |
| fit | priority_minus_warm | 0.380780 | 0.354046 | 0.407515 |
| held | priority_minus_affine | -1.291029 | -2.191849 | -0.390209 |
| held | priority_minus_continued | -0.568268 | -0.814705 | -0.321830 |
| held | priority_minus_warm | -0.909470 | -1.344157 | -0.474783 |

Prospective expansion condition: False.
Full original priority risk/support pass: False.
Worst defined signed recording easy-net degradation: 22.669832%.
No favorable context is chosen as a replacement result. See all fixed
contexts and admission diagnostics in [failure analysis](failure_analysis.md).

Seven already exposed SOURCE units, 17 recording views, one producer seed (17).
Completion bounds address missing labels, not sampling confidence intervals.
The original 24-group primary stays NA, not passed. Original 2% harm, easy-net and
support rules are unchanged. Empty matched controls remain unavailable;
no identity certificate. Deployment and submission readiness remain false.

[All metrics](analysis.json), [actual weight audit](fit_audit.json),
[protocol](protocol.md), [reproducibility](reproducibility.md).
Native obs8/pred12 image pixels, not SDD t50, verified metric/seconds,
human gold, true 3D or foundation evidence. Stage5C and SMC remain disabled.
