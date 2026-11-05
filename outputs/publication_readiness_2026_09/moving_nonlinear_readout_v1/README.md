# Nonlinear Risk Readout: Matched Representation Control

I tested whether frozen neural features contain risk information that the
previous affine output missed. Both arms use standard histogram gradient
boosting: score_only sees two signed risk scores; latent_score sees those
scores plus the same 64 hidden features. The intervention changes the risk
model, not the trajectory, policy thresholds or evaluation contract.

## Evidence Scope

fresh_run: six two-output readout fits, 1,536 scalar boosting rounds; real
resume pilot, one complete independent refit, inference/readout and replay.
cached_verified: neural encoders, SVD readouts, proposal router, source roles,
causal arrays, original forecasts and all parent policies.
not_run: new encoder/trajectory training, independent H, new seeds, the full
108-direction study or deployment. This is not new Torch world-model training.

Training phase: 39.06 seconds, 5.662 GB
peak RSS. Native arm64, single process; numerical thread pools are limited to
four. All six models freeze before held-source evaluation.
The first evaluation attempt failed on a nested-preprocessor metadata lookup.
The [explicit interface amendment](evaluation_interface_repair.md) preserves
that failure and all fitted weights, and restores the unchanged exclusion check.
It does not revise a result, source role or scientific criterion.

## Fit And Held-Source Results

| Context | Arm | Rows / unknown bounds | Initial MSE | Final MSE | Fitting reduction % |
|---|---|---:|---:|---:|---:|
| eqmotion_fold1_seed17_critic0 | score_only | 5956/189 | 0.073996026 | 0.045065962 | 39.096780 |
| eqmotion_fold1_seed17_critic0 | latent_score | 5956/189 | 0.073996026 | 0.026199176 | 64.593807 |
| eqmotion_fold0_seed17_critic1 | score_only | 3882/167 | 0.043132388 | 0.030503796 | 29.278674 |
| eqmotion_fold0_seed17_critic1 | latent_score | 3882/167 | 0.043132388 | 0.023687401 | 45.082102 |
| eqmotion_fold0_seed17_critic2 | score_only | 39497/918 | 1.242664734 | 1.102706502 | 11.262751 |
| eqmotion_fold0_seed17_critic2 | latent_score | 39497/918 | 1.242664734 | 0.734844452 | 40.865430 |

| Policy | Known gain vs floor % | Hard gain % | Switch % | Full risk/support | All violations | Easy violations | Undefined selected easy |
|---|---:|---:|---:|---:|---:|---:|---:|
| conditioned_veto | 5.331845 | 6.532826 | 31.794134 | 1/17 | 14 | 16 | 0 |
| mean_veto | 4.091202 | 5.035472 | 21.307681 | 0/17 | 13 | 15 | 0 |
| quantile90_veto | 0.151484 | 0.218176 | 1.519389 | 2/17 | 10 | 9 | 4 |
| readout_veto | 4.323320 | 5.354744 | 22.011498 | 0/17 | 13 | 15 | 0 |
| score_only_veto | 1.584545 | 1.917687 | 8.056982 | 0/17 | 13 | 15 | 2 |
| latent_score_veto | 1.509024 | 1.826066 | 8.461942 | 0/17 | 13 | 15 | 2 |

| Scope | Contrast | Known pp | Completion lower pp | Completion upper pp |
|---|---|---:|---:|---:|
| fit | latent_minus_score | 1.724345 | 1.720089 | 1.728601 |
| fit | latent_score_minus_readout | -2.673512 | -2.746315 | -2.600710 |
| fit | score_only_minus_readout | -4.397857 | -4.470065 | -4.325650 |
| held | latent_minus_score | -0.075521 | -0.180675 | 0.029634 |
| held | latent_score_minus_readout | -2.814296 | -3.828902 | -1.799689 |
| held | score_only_minus_readout | -2.738775 | -3.754842 | -1.722708 |

These are source-macro descriptive effects on 7 exposed SOURCE
units, 7 source views and 17 recording views with one producer
seed. Completion lower/upper bounds are not confidence intervals. No CI,
independent significance or multi-seed stability is claimed for this pilot.

The prespecified compute-allocation condition fails. Do not promote this variant or select a favorable context.
Full original latent-policy risk/support: False.
Worst defined signed recording easy-net degradation:
8.927122%, not a mean risk ratio.
Missing nonempty matched source views:
score_only=0,
latent_score=0.
The old 24-group primary remains NA, not passed. No identity-fallback
certification, retrospective success or removed failed view.

[Failure analysis](failure_analysis.md), [full numeric results](analysis.json),
[fixed protocol](protocol.md), [fit audit](fit_audit.json), [reproduction](reproducibility.md).
Deployment and submission readiness remain unchanged/false. Native obs8/pred12
image pixels, not SDD t50, verified seconds, metric, human-gold, true 3D or
foundation evidence. Stage5C and SMC remain disabled.
