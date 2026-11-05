# Causal Kinematic Risk Representation

I compared explicit backward motion, turn and neighbor-approach features with
an otherwise identical continued-training control. Both start from the same
neural weights; the new branch starts at zero, preserving initial predictions.
All common encoder weights learn. The masked arm receives no new fields.
The motion information already exists in the flat input; this tests its
representation, not a newly acquired modality or a new trajectory model.

## Execution And Evidence Scope

fresh_run: six real Torch heads, 12,000 primary updates, another 2,000-update
independent refit, 400-update real resume pilot, inference and exact replay.
Training took 491.91 seconds with 7.535 GB
peak RSS, native arm64 CPU4/interop1/workers0. All six fits precede evaluation.
cached_verified: forecast bank, router, prior neural weights, original source
roles, preprocessing, targets and parent control predictions/metrics.
not_run: new trajectory training, new independent H, new producer seeds, full
108-direction expansion or deployment. One-seed exposed SOURCE is development.

## Full Fitting Loss

These are fresh full fitting-data losses, using the exact empirical query
weights and original bound-aware normalized targets. They are not single
minibatch scores, validation metrics or uncertainty calibration.

| Context | Arm | Warm MSE | Final MSE | Reduction % | Known / bound-only proposals |
|---|---|---:|---:|---:|---:|
| eqmotion_fold1_seed17_critic0 | masked | 0.074977281 | 0.048837922 | 34.863039 | 5767/189 |
| eqmotion_fold1_seed17_critic0 | kinematic | 0.074977281 | 0.020248872 | 72.993323 | 5767/189 |
| eqmotion_fold0_seed17_critic1 | masked | 0.045257559 | 0.028839503 | 36.276938 | 3715/167 |
| eqmotion_fold0_seed17_critic1 | kinematic | 0.045257559 | 0.018960970 | 58.104303 | 3715/167 |
| eqmotion_fold0_seed17_critic2 | masked | 1.279164477 | 1.095263992 | 14.376610 | 38579/918 |
| eqmotion_fold0_seed17_critic2 | kinematic | 1.279164477 | 0.893710541 | 30.133258 | 38579/918 |

## Held SOURCE Outcomes

| Policy | Known floor gain % | Hard gain % | Switch % | Full risk/support | All/easy violations | Undefined selected easy |
|---|---:|---:|---:|---:|---|---:|
| conditioned_veto | 5.331845 | 6.532826 | 31.794134 | 1/17 | 14/16 | 0 |
| mean_veto | 4.091202 | 5.035472 | 21.307681 | 0/17 | 13/15 | 0 |
| quantile90_veto | 0.151484 | 0.218176 | 1.519389 | 2/17 | 10/9 | 4 |
| readout_veto | 4.323320 | 5.354744 | 22.011498 | 0/17 | 13/15 | 0 |
| masked_veto | 3.962463 | 5.125171 | 18.506051 | 0/17 | 13/16 | 0 |
| kinematic_veto | 3.941761 | 4.858158 | 17.315501 | 0/17 | 14/14 | 2 |

| Scope | Contrast | Known pp | Completion lower pp | Completion upper pp |
|---|---|---:|---:|---:|
| fit | kinematic_minus_affine | -0.310783 | -0.386196 | -0.235370 |
| fit | kinematic_minus_masked | -0.528802 | -0.589836 | -0.467769 |
| fit | masked_minus_mean | 0.948215 | 0.898700 | 0.997730 |
| held | kinematic_minus_affine | -0.381559 | -1.143891 | 0.380773 |
| held | kinematic_minus_masked | -0.020703 | -0.943057 | 0.901652 |
| held | masked_minus_mean | -0.128739 | -0.685339 | 0.427862 |

The prespecified expansion condition fails. No favorable context is selected for promotion.
Full original kinematic risk/support pass: False.
Worst defined signed recording easy-net degradation:
37.345112%.

Results are source-macro descriptive effects on 7 already exposed
SOURCE units, 7 source views, 17 recording views and seed17.
Completion bounds address missing labels, not sampling confidence intervals.
No CI or independent significance is claimed. The original 24-group primary
remains NA, not passed; empty controls remain unavailable and no identity-
fallback certificate is added. Original 2% harm, easy-net and support rules
are unchanged. Deployment and submission readiness are unchanged/false.

[Failure and mechanism analysis](failure_analysis.md), [input trace](input_diagnosis.md),
[protocol](protocol.md), [all metrics](analysis.json), [weight audit](fit_audit.json),
[reproduction](reproducibility.md).

Native obs8/pred12 image pixels, not SDD t50, verified metric/seconds,
human-gold, true3D or foundation evidence. Stage5C and SMC remain disabled.
