# Separate Risk And Decision Representations

I trained three neural architectures in each of three own-fitting contexts
under the unchanged 2% all/easy risk and support rules. The separated model
learns risk magnitude and intervention decisions in independent encoders.
Narrow and near-capacity-matched wide shared encoders are fixed controls.
All nine final iterates are reported; none is selected using held outcomes.

## Complete Fitting Results

| Context | Architecture | Parameters | Mean MSE reduction % | Easy MSE reduction % | All/easy decision-cost reduction % |
|---|---|---:|---:|---:|---:|
| eqmotion_fold1_seed17_critic0 | shared64 | 32260 | 7.286 | 6.794 | -8.971 / -38.326 |
| eqmotion_fold1_seed17_critic0 | shared128 | 64516 | 18.713 | 18.467 | -0.540 / 4.610 |
| eqmotion_fold1_seed17_critic0 | separate64 | 64260 | 20.417 | 20.717 | -4.128 / 20.538 |
| eqmotion_fold0_seed17_critic1 | shared64 | 32260 | 18.167 | 20.269 | 15.226 / 73.105 |
| eqmotion_fold0_seed17_critic1 | shared128 | 64516 | 27.848 | 28.495 | 38.464 / 72.880 |
| eqmotion_fold0_seed17_critic1 | separate64 | 64260 | 28.682 | 32.034 | 30.712 / 75.252 |
| eqmotion_fold0_seed17_critic2 | shared64 | 32260 | 6.450 | 6.467 | 18.336 / 85.206 |
| eqmotion_fold0_seed17_critic2 | shared128 | 64516 | 8.953 | 8.803 | 19.167 / 84.693 |
| eqmotion_fold0_seed17_critic2 | separate64 | 64260 | 10.068 | 10.114 | 20.184 / 84.964 |

Positive means improvement on fitting data versus its warm initialization.
These are normalized risk-regression losses and observed-label decision costs,
NOT trajectory ADE/FDE, test gains, or safety probabilities. Conservative
missing-label supervision remains active in both heads. Training all/easy
cost weighting is unchanged. A fall in MSE alone does not pass the screen.

## Capacity-Controlled Comparison

| Context | Separate64 versus | MSE reduction % | All/easy decision-cost reduction % |
|---|---|---:|---:|
| eqmotion_fold0_seed17_critic1 | shared128 | 1.156 | -12.598 / 8.749 |
| eqmotion_fold0_seed17_critic1 | shared64 | 12.849 | 18.267 / 7.982 |
| eqmotion_fold0_seed17_critic2 | shared128 | 1.224 | 1.258 / 1.770 |
| eqmotion_fold0_seed17_critic2 | shared64 | 3.867 | 2.263 / -1.636 |
| eqmotion_fold1_seed17_critic0 | shared128 | 2.097 | -3.569 / 16.697 |
| eqmotion_fold1_seed17_critic0 | shared64 | 14.163 | 4.444 / 42.555 |

Wide shared has256 more parameters than separated (64,516 versus64,260).
The primary comparison is against this wide control; the narrow control has
32,260 parameters. Initialization is function-matched within numerical
tolerance, not identical parameterization. Shared/separate gradients differ.
This is not a proof of capacity-independent superiority or architectural novelty.

## Development Readout

Prospective fitting expansion screen: **False**.
**not_run:** Prespecified separate64 fitting screen failed; no additional V or H predictions.
No new held utility or safety result is inferred from fitting improvements.
Earlier held failures remain failures. The model is not deployed.

## Computation And Replay

18,000 primary updates, 2,000 exact full-refit updates, and600 pilot updates
including repeats. Each fit uses the same2,000 query draws as its original
constant control. All9 initial/final complete-fitting metrics replay exactly.
All3 arms pass50+50 versus100 actual-update resume and prediction equality.
The first separate64 full fit repeats exactly. A budget cap is not convergence.

| Phase | PID | Seconds | Peak RSS bytes |
|---|---:|---:|---:|
| pilot | 95467 | 84.82 | 6640500736 |
| train | 95634 | 1059.52 | 7916552192 |
| evaluate | 97225 | 16.83 | 1014120448 |
| verify | 97261 | 32.24 | 5872599040 |

A completed evaluate command can only record not_run; it is not evidence
that held predictions ran. Real arm64 PyTorch CPU4/inter-op1/workers0 was
used, not NumPy fallback. Checkpoints remain private and hash-bound.

Original24-group primary remains NA, not passed. Empty controls/missing
support remain NA; no identity certification or new H/reserved readout.
This is an intervention-risk model, not newly trained trajectory dynamics.
Native obs8/pred12 image coordinates; no metric/seconds/true3D/foundation
or human-gold claim. Stage5C and SMC remain disabled.

[Protocol](protocol.md) | [Numeric summary](result_summary.json) | [Analysis](analysis.json)
[Interpretation](result_interpretation.md) | [Recovery guide](operation_zh.md)
