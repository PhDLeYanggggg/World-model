# Frozen Neural Readout: Optimization Control

I kept the original evaluation rules and changed only the two affine risk
outputs of three frozen neural representations. Weighted float64 SVD refits
130 parameters per model using the original empirical query weights. Actual
saved float32 inference is checked separately. This is a standard optimization
control, not a new trajectory model, loss invention or safety certificate.

## Evidence Scope

- fresh_run: three real-data output solves, one independent complete repeat,
  causal inference, original policy readout and exact independent replay.
- cached_verified: original neural encoders, forecasts, source roles, causal
  features, source-specific preprocessing, proposal support and parent policies.
- not_run: new encoder training, full 108-context-direction comparison,
  additional seeds, independent H confirmation or deployment.

Training phase: 25.94 seconds, peak RSS
5.647 GB. There were zero new Adam updates.
All three output systems have retained rank 65; encoders remain bitwise fixed.
This pilot has 7 SOURCE units, 7 source views,
17 recording views and one producer seed. These are already exposed development
sources, not independent confirmation. No CI or multi-seed effect is claimed.

## Fitting Loss And Original Policy Evaluation

| Frozen context | Before MSE | After MSE | Reduction % | Rank | Relative normal residual |
|---|---:|---:|---:|---:|---:|
| eqmotion_fold1_seed17_critic0 | 0.074977281 | 0.073996026 | 1.308738 | 65/65 | 1.034e-14 |
| eqmotion_fold0_seed17_critic1 | 0.045257559 | 0.043132388 | 4.695727 | 65/65 | 4.154e-15 |
| eqmotion_fold0_seed17_critic2 | 1.279164477 | 1.242664734 | 2.853405 | 65/65 | 5.214e-15 |

| Policy | Known floor gain % | Hard gain % | Switch % | Full support | All violations | Easy violations | Undefined selected easy |
|---|---:|---:|---:|---:|---:|---:|---:|
| public_router | 14.496012 | 18.306013 | 62.383468 | 0/17 | 15 | 17 | 0 |
| conditioned_veto | 5.331845 | 6.532826 | 31.794134 | 1/17 | 14 | 16 | 0 |
| mean_veto | 4.091202 | 5.035472 | 21.307681 | 0/17 | 13 | 15 | 0 |
| quantile90_veto | 0.151484 | 0.218176 | 1.519389 | 2/17 | 10 | 9 | 4 |
| readout_veto | 4.323320 | 5.354744 | 22.011498 | 0/17 | 13 | 15 | 0 |

| Scope, readout minus mean | Known pp | Completion lower pp | Completion upper pp |
|---|---:|---:|---:|
| fit | 0.730195 | 0.714314 | 0.746077 |
| held | 0.232118 | 0.048443 | 0.415793 |

The prespecified allocation condition is not met. Do not promote this output-only repair.
Full original risk/support pass: False. Worst defined
signed recording easy-net degradation: 39.884529%.
The nonempty matched comparison is unavailable in
0/7 source views.
The original 24-group primary remains NA, not passed; these pilot tables do
not replace it. No identity-fallback certification is introduced.

See [failure analysis](failure_analysis.md), [complete results](analysis.json),
[fit audit](fit_audit.json), [gates](gates.md) and [reproduction](reproducibility.md).
Deployment and submission readiness remain false. Native obs8/pred12 image
pixels are not verified seconds, metric units, SDD t50, true 3D or foundation
evidence. Labels are not human gold. Stage5C and SMC remain disabled.
