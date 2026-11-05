# Learning The Composed Intervention Decision

I tested whether directly training the conjunction of two neural decisions
corrects the false rejections seen with independently trained heads. Both
arms retain the original risk-regression and conservative missing-label
losses. The treatment adds an observed-label cost-sensitive loss on their
per-channel maximum. The 2% risk, easy-net and support rules are unchanged.

## Fixed Matched Comparison

Six actual fits use the same three fitting contexts, 64,260 parameters,
492 causal inputs, warm initialization, query draws and 2,000 updates.
All three fresh independent controls exactly reproduce the previous
separated models, optimizer states and predictions. No checkpoint or
threshold is selected from held outcomes. The final treatment is repeated
exactly for the first context. A finite update budget is not convergence.

| Context | Loss | Risk MSE reduction % | All/easy decision-cost reduction % |
|---|---|---:|---:|
| eqmotion_fold1_seed17_critic0 | independent | 20.417 | -4.128 / 20.538 |
| eqmotion_fold1_seed17_critic0 | composed | 19.599 | 28.091 / 31.380 |
| eqmotion_fold0_seed17_critic1 | independent | 28.682 | 30.712 / 75.252 |
| eqmotion_fold0_seed17_critic1 | composed | 28.261 | 40.883 / 77.593 |
| eqmotion_fold0_seed17_critic2 | independent | 10.068 | 20.184 / 84.964 |
| eqmotion_fold0_seed17_critic2 | composed | 9.927 | 28.078 / 85.142 |

Positive means improvement on fitting data versus its warm model.
These are risk-regression losses and observed sign costs, NOT trajectory
ADE/FDE or test improvements. Unknown labels keep their original bounds.

| Context | Composed versus independent: MSE reduction % | All/easy cost reduction % |
|---|---:|---:|
| eqmotion_fold0_seed17_critic1 | -0.591 | 14.679 / 9.457 |
| eqmotion_fold0_seed17_critic2 | -0.157 | 9.890 / 1.186 |
| eqmotion_fold1_seed17_critic0 | -1.028 | 30.942 / 13.644 |

Internal fitting expansion screen: **3/3 contexts**, overall **True**.
All three must reduce mean risk MSE by at least1% and not increase either
observed combined sign cost. This is an internal readout screen, not the
scientific safety gate. The matched loss contrast is also reported above.

## Held Development Readout

Both arms and risk-only diagnostics were frozen before evaluation
on the same seven exposed SOURCE units /17 recording views.

| Policy | Known ADE gain % | Hard gain % | Switch % | Full original risk/support | Worst easy-net % |
|---|---:|---:|---:|---:|---:|
| constant_veto | 3.601 | 4.626 | 15.014 | 0/17 | 38.905 |
| independent_veto | 0.745 | 0.995 | 5.231 | 0/17 | 8.116 |
| composed_veto | 0.759 | 1.009 | 5.294 | 0/17 | 8.038 |
| independent_risk_only_veto | 3.287 | 4.252 | 14.401 | 0/17 | 38.220 |
| composed_risk_only_veto | 3.315 | 4.287 | 14.596 | 0/17 | 40.378 |

| Contrast | Known pp | Completion lower pp | Completion upper pp |
|---|---:|---:|---:|
| composed_decision_head | -2.556 | -3.769 | -1.343 |
| independent_decision_head | -2.541 | -3.747 | -1.335 |
| main | 0.013 | 0.008 | 0.019 |
| vs_constant | -2.842 | -4.038 | -1.646 |

Completion bounds are not sampling confidence intervals. Partial
known ADE is not complete-roster12-step ADE. Cached controls are
hash-verified with exact fresh metric recomputation. Repeated exposed
development is not independent confirmation or a three-seed primary.

## Error Attribution

| Context | Loss | All false-safe cost reduction % | All false-reject cost reduction % | Easy false-safe cost reduction % | Easy false-reject cost reduction % |
|---|---|---:|---:|---:|---:|
| eqmotion_fold1_seed17_critic0 | independent | 75.821 | -20.545 | 91.796 | -34.977 |
| eqmotion_fold1_seed17_critic0 | composed | 82.874 | 16.842 | 95.875 | -18.867 |
| eqmotion_fold0_seed17_critic1 | independent | 87.847 | 11.472 | 95.860 | -74.255 |
| eqmotion_fold0_seed17_critic1 | composed | 92.533 | 23.490 | 98.473 | -73.893 |
| eqmotion_fold0_seed17_critic2 | independent | 93.480 | -45.055 | 99.691 | -61.873 |
| eqmotion_fold0_seed17_critic2 | composed | 93.535 | -30.185 | 99.891 | -61.905 |

All per-source decompositions are in fitting_error_attribution.json.
These analyses diagnose the fixed models; they do not select a stand-alone
branch or erase failures. The maximum trains per-channel sign conjunction,
not the full population risk ratio or reference support. Its magnitudes
are not calibrated probabilities or certified upper risk bounds.

## Runtime And Evidence Scope

| Phase | PID | Seconds | Peak RSS bytes |
|---|---:|---:|---:|
| pilot | 98622 | 64.70 | 6617432064 |
| train | 98758 | 938.05 | 6777159680 |
| evaluate | 1815 | 66.75 | 5588910080 |
| verify | 2026 | 68.93 | 5683314688 |
| attribution | 2273 | 33.21 | 5695946752 |
| attribution_replay | 2488 | 33.12 | 5739184128 |

12,000 primary updates +2,000 exact full-refit updates +400 pilot updates.
Real native arm64 PyTorch CPU4/inter-op1/workers0, not NumPy fallback.
All six initial/final fitting metrics and error decompositions replay exactly.
Private model/data files remain excluded from Git; only light receipts are public.

| Evidence | Origin | Meaning |
|---|---|---|
| Data/schema/partitions/warm starts | cached_verified | Inherited bindings checked, not new collection |
| Six model fits and fitting attribution | fresh_run | Real training and fresh computation |
| Independent control match | fresh_run plus cached_verified | Fresh fit compared with frozen previous artifact |
| Additional nested-C development | fresh_run_exposed_development | No independent-test claim |
| Reserved/H confirmation | not_run | Remains closed |

No deployment upgrade or submission-ready claim. The original24-group
primary remains NA, not passed; empty controls and missing support stay NA.
No identity certification or new H/reserved Moving/DUT/DroneCrowd reading.
This trains intervention-risk heads, not new JEPA or trajectory dynamics.
Native obs8/pred12 image coordinates; no metric, seconds, true3D, foundation
or human-gold claim. Stage5C and SMC remain disabled.

[Protocol](protocol.md) | [Numeric summary](result_summary.json)
[Interpretation](result_interpretation.md) | [Recovery](operation_zh.md)
