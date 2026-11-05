# Explicit Scale Removal: Completed Neural Ablation

I remove the risk head's explicit absolute-scale input while retaining
normalized geometry and the original losses, source roles and risk rules.
Six fresh continuations are compared with six cached_verified unmasked
controls. Main contrast is MSE; signed-cost learning remains secondary.

## Held Development Results

| Model | Known ADE gain over floor % | Hard gain % | Intervention % | Full risk/support |
|---|---:|---:|---:|---:|
| readout_veto | 4.3233 | 5.3547 | 22.0115 | 0/17 |
| kinematic_veto | 3.9418 | 4.8582 | 17.3155 | 0/17 |
| continued_mse_veto | 3.6006 | 4.6257 | 15.0144 | 0/17 |
| ablated_continued_mse_veto | 4.1531 | 5.4549 | 17.6952 | 0/17 |
| signed_cost_logistic_veto | 2.6928 | 3.5649 | 17.6705 | 1/17 |
| ablated_signed_cost_logistic_veto | 3.0085 | 4.3291 | 18.2195 | 1/17 |

| Objective | Ablated minus unmasked known pp | Completion lower pp | Completion upper pp | Expansion screen |
|---|---:|---:|---:|---|
| continued_mse | 0.5525 | 0.0944 | 1.0106 | False |
| signed_cost_logistic | 0.3158 | 0.1141 | 0.5174 | False |

Missing-label completion bounds are NOT sampling confidence intervals.
An expansion screen is not a safety certificate.

## Original Safety Accounting

| Variant | Known all violations | Known easy violations | Undefined selected easy | Worst defined easy-net % |
|---|---:|---:|---:|---:|
| continued_mse_veto | 14 | 13 | 2 | 38.9046 |
| ablated_continued_mse_veto | 13 | 13 | 2 | 47.5913 |
| signed_cost_logistic_veto | 11 | 14 | 1 | 26.2254 |
| ablated_signed_cost_logistic_veto | 11 | 14 | 2 | 30.2521 |

Original24group primary remainsNA, notpassed. Empty utility controls and
missing supports stayNA. No identity certificate, new test, deployment
or submission-readiness upgrade. Prior failures remain unchanged.

## Runtime

| Phase | PID | Seconds | Peak RSS bytes |
|---|---:|---:|---:|
| pilot | 87463 | 37.04 | 2850275328 |
| train | 87626 | 521.02 | 9240248320 |
| evaluate | 88461 | 48.02 | 4613537792 |
| verify | 88560 | 46.73 | 4514856960 |

Nativearm64Torch CPU4/interop1/workers0. Six fits12,000primaryupdates,
2,000fullrefitupdates,400resume-checkupdates. All six freeze before V.
Both objective arms resume exactly. First masked logistic full refit
and all new predictions/actions/metrics replay exactly. Old actions are
hash-verified; old metrics recomputed fresh and match exactly.

## Limits

Seven exposed SOURCE units,17recording views,one producerseed. Repeated
development evidence, not independent confirmation. Overlapping windows
are not additional independent scenes. B has only2/3/3sourcegroups.
The frozen proposer and support models retain original scale information;
warm weights previously saw scale. This is not full-policy invariance.
No future inference inputs, new trajectory dynamics, JEPA or reservedH.
Nativeobs8/pred12 imagepixels: no verifiedseconds/metric/true3D/foundation
or human-gold claim. Stage5C/SMCoff. Original2%and support rules unchanged.

[Protocol](protocol.md) | [Fitting audit](full_fitting_objectives.json) | [Replay](verification.json)
