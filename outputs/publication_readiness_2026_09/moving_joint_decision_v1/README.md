# Learning The Complete Intervention Decision

I tested one loss repair after a fitting-only audit found that zero-target
channels can block useful proposals despite receiving no signed-cost
gradient. The treatment trains the complete four-output conjunction.
Risk MSE, the original decision objective, conservative missing-label
supervision and the original 2% risk/easy-net/support rules are unchanged.

## Controlled Experiment

Six real neural fits compare per-channel and joint auxiliary losses in
three own-B contexts. All use 64,260 parameters, 492 causal inputs, identical
warm starts, query draws and 2,000 updates. The three fresh controls exactly
reproduce the preceding composed models, optimizers and predictions.
The first full joint fit and both pilot resume paths reproduce exactly.

| Context | Loss | MSE reduction from warm % | All/easy sign-cost reduction from warm % |
|---|---|---:|---:|
| eqmotion_fold1_seed17_critic0 | per_channel | 19.599 | 28.091 / 31.380 |
| eqmotion_fold1_seed17_critic0 | joint | 19.873 | 19.054 / 36.039 |
| eqmotion_fold0_seed17_critic1 | per_channel | 28.261 | 40.883 / 77.593 |
| eqmotion_fold0_seed17_critic1 | joint | 27.820 | 36.617 / 76.913 |
| eqmotion_fold0_seed17_critic2 | per_channel | 9.927 | 28.078 / 85.142 |
| eqmotion_fold0_seed17_critic2 | joint | 9.897 | 26.449 / 84.477 |

These are fitting diagnostics, not trajectory ADE/FDE or test gains.

| Context | Matched MSE reduction % | All/easy cost reduction % | Joint cost reduction % | False-reject reduction % | False-safe reduction % | Zero-only reject reduction % |
|---|---:|---:|---:|---:|---:|---:|
| eqmotion_fold0_seed17_critic1 | -0.614 | -7.216 / -3.034 | 17.585 | 19.162 | -138.728 | 52.548 |
| eqmotion_fold0_seed17_critic2 | -0.033 | -2.265 / -4.474 | 30.113 | 31.934 | -1303.028 | 67.356 |
| eqmotion_fold1_seed17_critic0 | 0.341 | -12.567 / 6.790 | 13.903 | 14.001 | 3.961 | 45.325 |

Original internal expansion screen: **3/3**, overall **True**.
Each treatment context must improve risk MSE at least 1% from warm and
not worsen either channel sign cost. This is not a scientific safety gate.
Joint attribution is diagnostic; it cannot replace a failed screen.

## Joint Error Attribution

| Context | Loss | Target-safe proposals | False rejections | False-safe proposals | Zero-only rejection cost | Nonzero-only cost | Both cost |
|---|---|---:|---:|---:|---:|---:|---:|
| eqmotion_fold1_seed17_critic0 | per_channel | 5303 | 2987 | 16 | 0.00288060 | 0.00166763 | 0.00288125 |
| eqmotion_fold1_seed17_critic0 | joint | 5303 | 2655 | 18 | 0.00157498 | 0.00156282 | 0.00325149 |
| eqmotion_fold0_seed17_critic1 | per_channel | 2248 | 2073 | 20 | 0.00167404 | 0.00039041 | 0.00196770 |
| eqmotion_fold0_seed17_critic1 | joint | 2248 | 1924 | 35 | 0.00079437 | 0.00056579 | 0.00189934 |
| eqmotion_fold0_seed17_critic2 | per_channel | 24216 | 23981 | 22 | 0.01361692 | 0.00151898 | 0.01292320 |
| eqmotion_fold0_seed17_critic2 | joint | 24216 | 22704 | 162 | 0.00444511 | 0.00210232 | 0.01255140 |

Zero-only/nonzero-only/both are disjoint. Per-output blocker counts
overlap and cannot be added as independent samples. Target-safe means
observed-label normalized sign only, not a deployment safety certificate.
Per-source attribution is retained in every fitting receipt.

| Context | Fitting source | Joint cost change | False-safe cost change | False-reject cost change |
|---|---|---:|---:|---:|
| eqmotion_fold0_seed17_critic1 | moving_source_component_053 | -0.00062772 | +0.00003602 | -0.00066374 |
| eqmotion_fold0_seed17_critic1 | moving_source_component_065 | -0.00005054 | +0.00000000 | -0.00005054 |
| eqmotion_fold0_seed17_critic1 | moving_source_component_076 | -0.00003793 | +0.00002045 | -0.00005838 |
| eqmotion_fold0_seed17_critic2 | moving_source_component_059 | -0.00832568 | +0.00034866 | -0.00867434 |
| eqmotion_fold0_seed17_critic2 | moving_source_component_060 | -0.00014010 | +0.00011924 | -0.00025934 |
| eqmotion_fold0_seed17_critic2 | moving_source_component_073 | +0.00000471 | +0.00003130 | -0.00002659 |
| eqmotion_fold1_seed17_critic0 | moving_source_component_048 | -0.00051124 | -0.00000406 | -0.00050718 |
| eqmotion_fold1_seed17_critic0 | moving_source_component_067 | -0.00053184 | +0.00000116 | -0.00053300 |

Positive changes above are worse. These are absolute contributions
under the same original query weighting, not source-standardized effects
or independent-sample estimates. Negative slices are retained.

## Held Development

Seven already exposed SOURCE units / 17 recording views, one producer
seed. Every model/action freezes before outcomes are accessed.

| Policy | Known ADE gain % | Hard gain % | Switch % | Worst defined easy-net % | Full original risk/support |
|---|---:|---:|---:|---:|---:|
| constant_veto | 3.601 | 4.626 | 15.014 | 38.905 | 0/17 |
| per_channel_veto | 0.759 | 1.009 | 5.294 | 8.038 | 0/17 |
| joint_veto | 0.895 | 1.170 | 5.988 | 8.633 | 0/17 |
| joint_risk_only_veto | 3.317 | 4.290 | 14.562 | 40.378 | 0/17 |

| Contrast | Known pp | Completion lower pp | Completion upper pp |
|---|---:|---:|---:|
| joint_decision_head | -2.422 | -3.589 | -1.254 |
| main | 0.136 | 0.093 | 0.180 |
| per_channel_decision_head | -2.556 | -3.769 | -1.343 |
| vs_constant | -2.706 | -3.864 | -1.547 |

The joint policy has 13/17 known all-risk violations, 13/17 known easy-risk violations, and 3 undefined selected-easy views.
**Full original risk/support pass: False. No deployment upgrade.**

Completion bounds are not sampling CIs. Known ADE uses available
outcomes, not complete-roster 12-step accuracy. Repeated exposed
development is not independent confirmation or multi-seed evidence.

## Execution And Evidence

| Phase | PID | Seconds | Peak RSS bytes |
|---|---:|---:|---:|
| pilot | 4791 | 74.96 | 6492389376 |
| train | 4934 | 1051.49 | 7558004736 |
| evaluate | 6819 | 63.68 | 5575016448 |
| verify | 7216 | 69.12 | 6397788160 |

12,000 primary updates, 2,000 exact full-refit updates, 400 pilot updates.
Real arm64 PyTorch CPU 4/inter-op 1/workers 0, not NumPy fallback. All six
initial/final metrics and joint attributions replay exactly. Models,
optimizers, sampler and RNG are checkpointed every 100 completed updates.

Parent data/schema/warm assets: cached_verified. Training and metrics:
fresh_run. New reserved/H confirmation: not_run. Private arrays and weights
remain excluded from Git. Hashes alone are not a self-contained benchmark.

No deployment or submission-ready claim. Original 24-group primary remains
NA, not passed; empty controls and missing support remain NA. No identity
certification, threshold change or reserved Moving/DUT/DroneCrowd/H reading.
Neural intervention-risk training, not new JEPA or forecasting dynamics.
Native obs8/pred12 image coordinates, not metric/seconds/true 3D/foundation/
human-gold evidence. Stage5C and SMC disabled.

[Protocol](protocol.md) | [Numeric summary](result_summary.json) |
[Failure analysis](failure_analysis.md) | [Recovery guide](operation_zh.md)
