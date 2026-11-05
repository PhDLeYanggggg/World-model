# Fixed-Schedule Neural Risk Training

I compare constant and gradually reduced learning rates while retaining
the original model, causal inputs, labels, loss, update budget and safety
rules. Six real neural fits and a full treatment refit are complete.
No thresholds or evaluation support rules were changed.

## Held Development Evidence

| Model | Known ADE gain over floor % | Hard gain % | Intervention % | Full risk/support | Worst defined easy-net % |
|---|---:|---:|---:|---:|---:|
| readout_veto | 4.3233 | 5.3547 | 22.0115 | 0/17 | 39.8845 |
| kinematic_veto | 3.9418 | 4.8582 | 17.3155 | 0/17 | 37.3451 |
| constant_veto | 3.6006 | 4.6257 | 15.0144 | 0/17 | 38.9046 |
| cosine_veto | 3.7549 | 4.7483 | 16.5645 | 0/17 | 44.6540 |

| Contrast | Known pp | Completion lower pp | Completion upper pp |
|---|---:|---:|---:|
| cosine_minus_affine | -0.5685 | -1.4704 | 0.3335 |
| cosine_minus_constant | 0.1543 | 0.0275 | 0.2811 |

**Completion bounds are not sampling confidence intervals.**
These are repeated development results on seven SOURCE units and one
producer seed, not independent confirmation. Partial observed ADE is not
complete-roster 12-step ADE. The original 24-group primary remains NA.

## Original Safety Checks

| Schedule | Known all-risk violations | Known easy-risk violations | Undefined selected easy | Undefined full easy |
|---|---:|---:|---:|---:|
| constant | 14 | 13 | 2 | 0 |
| cosine | 14 | 15 | 0 | 0 |

Development expansion screen: False.
Full recording risk/support gate: False.
A local improvement cannot replace the full safety gate. Empty controls
and missing support remain NA, without an identity-fallback certificate.

## Whole-Fitting Diagnostics

Computed before held prediction, over all own-B queries with the same
empirical sampling weights. Lower training loss is not held-scene safety.

These losses measure normalized signed budget-excess risk prediction,
not trajectory ADE/FDE. Unknown outcomes retain conservative bound
supervision, rather than being silently dropped.

| Context | Schedule | All MSE | Easy MSE | All/easy sign regret | False-safe all/easy |
|---|---|---:|---:|---|---|
| eqmotion_fold1_seed17_critic0 | constant | 0.0041519294 | 0.019720747 | [0.0047020691156016954, 0.0016522157767175723] | [127, 76] |
| eqmotion_fold1_seed17_critic0 | cosine | 0.0051791693 | 0.025291614 | [0.005363824212208592, 0.0017924289143319908] | [95, 58] |
| eqmotion_fold0_seed17_critic1 | constant | 0.0041351828 | 0.019137472 | [0.0039158054996790885, 0.0004650252474167701] | [225, 73] |
| eqmotion_fold0_seed17_critic1 | cosine | 0.004437153 | 0.021781604 | [0.0035603797372873855, 0.000580875026777525] | [284, 80] |
| eqmotion_fold0_seed17_critic2 | constant | 0.19822064 | 1.1618271 | [0.019497021051193053, 0.006812996655288054] | [2804, 952] |
| eqmotion_fold0_seed17_critic2 | cosine | 0.2029645 | 1.2140707 | [0.02006812411545289, 0.007069800453269129] | [2967, 1044] |

## Runtime And Reproducibility

| Phase | PID | Seconds | Peak RSS bytes |
|---|---:|---:|---:|
| pilot | 90244 | 35.73 | 2834153472 |
| train | 90371 | 494.94 | 7561248768 |
| evaluate | 91232 | 47.37 | 4502257664 |
| verify | 91354 | 46.37 | 4539645952 |

Native arm64 Torch CPU4/inter-op1/workers0. Both schedules resume
exactly; the first full cosine refit matches exactly. Three fresh constant
controls reproduce old weights, optimizer and sample draws. All new held
predictions, actions and metrics replay exactly; old cached actions are
hash-verified and their metrics recomputed. All six fits froze before V.
These are engineering and development checks, not independent test results.

No new H/reserved data, deployment upgrade, trajectory-dynamics training
or JEPA pretraining. Native obs8/pred12 image pixels: no metric, seconds,
true3D, foundation or human-gold claim. Stage5C and SMC remain disabled.

[Protocol](protocol.md) | [Analysis](analysis.json) | [Replay](verification.json)
