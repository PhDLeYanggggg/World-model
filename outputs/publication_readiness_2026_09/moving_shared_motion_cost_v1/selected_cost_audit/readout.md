# Selected-Cost Reliability Diagnosis

This is a post-readout diagnostic, not a preregistered new test or
independent confirmation. All72job/role/feature views replay exactly.
No fitting, policy selection, threshold or source role changes.

Each row pools18repeated job views. These are mass ratios, not official
equal-source-group averages or independent sample counts. Future labels
are used only after the frozen causal choices, to diagnose their errors.
Known H/EH calibration compares actual and predicted masses on the same
known rows; prediction budgets use all causally selected rows.

| Role | Arm | Policy | Predicted all/easy risk (%) | Known realized all/easy risk (%) | Known H / predicted H | Known EH / predicted EH |
|---|---|---|---:|---:|---:|---:|
| critic_fitting | local_history | all_neural | 14.863/188.345 | 14.882/47.464 | 1.467 | 0.646 |
| critic_fitting | local_history | independent | 0.156/0.052 | 9.679/31.684 | 64.461 | 653.069 |
| critic_fitting | shared_motion | all_neural | 22.102/297.276 | 14.882/47.464 | 1.172 | 0.466 |
| critic_fitting | shared_motion | independent | 0.147/0.031 | 10.026/33.168 | 76.756 | 1209.909 |
| held | local_history | all_neural | 38.543/1093.053 | 14.882/47.464 | 0.416 | 0.076 |
| held | local_history | independent | 0.124/0.047 | 11.453/35.599 | 75.719 | 1020.357 |
| held | local_history | joint_free | 0.354/0.604 | 11.658/36.996 | 28.051 | 91.030 |
| held | shared_motion | all_neural | 48.416/1041.334 | 14.882/47.464 | 0.429 | 0.081 |
| held | shared_motion | independent | 0.084/0.023 | 11.752/34.213 | 128.231 | 1783.199 |
| held | shared_motion | joint_free | 0.376/0.457 | 11.632/36.258 | 29.852 | 102.842 |

## Diagnosis

Global moment fit does not ensure reliability after selecting predictions
with very low estimated harm. The independent policy substantially
underpredicts selected harm already on critic-fitting sources; held
shift is not the sole explanation. Known-label violations also rule out
missing outcomes as the sole cause. This does not uniquely establish
whether sparse-tail weighting, decoder saturation, optimization or
insufficient causal information caused the conditional failure.

Do not add unrelated features or repeat a width/duration sweep from this
table alone. The next controlled test should first check selected-set
cost self-consistency on critic-fitting sources, retaining unknown bounds
and the original2%requirements. Any prospective training repair needs
a separate frozen definition; this diagnostic itself trains nothing.

The original per-recording support/risk failures remain. Pooling does not
rescue empty views or permit deployment. No new independent evidence,
metric/seconds,true3D/foundation,Stage5Cexecution or SMC claim.
