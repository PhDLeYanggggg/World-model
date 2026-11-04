# Matched Amplitude-Conditioned Cost Learning

## Material Passport

Fresh_run: 36 real Torch cost heads, 72,000 new primary updates, two
complete first-job refits, fitting-region checks and full policy inference.
Cached_verified: original forecasts/data and legacy binary policy metrics,
plus a separate exact replay of all new cost outputs and policy scalars.
The old forecasts are not retrained. These are exposed-source development
secondary results, not independent final confirmation. Original 24-group
primary remains not estimable; all 23 supported groups remain.

## Registered Contrasts

Joint-free amplitude-bank minus binary-only known: +0.4579 [-0.4805,+1.4930] pp.
Fixed paired completion lower: -1.1584 [-2.9064,+0.2188] pp.
Shared-weight full bank minus binary deployment known: +0.5947 [+0.2988,+0.9192] pp.
Shared-weight paired completion lower: +0.1010 [-0.4279,+0.5436] pp.

Intervals use 3,000 source-group resamples after excluded-producer and
actual-seed averaging. They are nominal development intervals, not
independent confirmation or adjustments for the wider research search.

| Model | Policy | Known gain over floor (%) [95% CI] | Completion lower (%) [95% CI] | Original risk support | Known all/easy violations | Undefined easy | Worst known easy-net degradation (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| amplitude_bank | floor | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] | 0/348 | 0/0 | 348 | 0.0000 |
| amplitude_bank | frame_uniform | +6.2459 [+2.8258,+10.3041] | +2.2958 [-2.4810,+6.3846] | 1/348 | 146/166 | 160 | 305.3475 |
| amplitude_bank | independent | +10.6394 [+4.7628,+16.7147] | +5.4507 [-3.4143,+12.9332] | 12/348 | 270/276 | 47 | 376.3311 |
| amplitude_bank | joint_free | +11.7276 [+5.1202,+18.6581] | +4.5682 [-4.4820,+12.0322] | 10/348 | 273/281 | 46 | 376.3311 |
| amplitude_bank | joint_matched | +11.6428 [+5.1058,+18.4680] | +4.5969 [-4.4231,+12.0125] | 7/348 | 276/281 | 50 | 376.3311 |
| amplitude_bank | uncontrolled_neural | +16.4074 [+7.4982,+25.5310] | -5.0187 [-16.4749,+5.0800] | 6/348 | 304/320 | 15 | 1742.9603 |
| amplitude_bank_binary_deploy | floor | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] | 0/348 | 0/0 | 348 | 0.0000 |
| amplitude_bank_binary_deploy | frame_uniform | +5.5309 [+2.3613,+9.3826] | +1.9666 [-2.5772,+5.9378] | 0/348 | 154/159 | 185 | 305.4969 |
| amplitude_bank_binary_deploy | independent | +9.7155 [+4.1072,+15.6715] | +4.7390 [-4.1976,+12.2020] | 7/348 | 280/268 | 66 | 428.9130 |
| amplitude_bank_binary_deploy | joint_free | +11.1330 [+4.5691,+18.0081] | +3.9187 [-5.4017,+11.6387] | 7/348 | 283/275 | 59 | 428.9130 |
| amplitude_bank_binary_deploy | joint_matched | +10.6500 [+4.3983,+17.2229] | +3.6957 [-5.2239,+11.0500] | 5/348 | 284/271 | 64 | 428.9130 |
| amplitude_bank_binary_deploy | uncontrolled_neural | +16.4074 [+7.4982,+25.5310] | -5.0187 [-16.4749,+5.0800] | 6/348 | 304/320 | 15 | 1742.9603 |
| binary_only | floor | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] | 0/348 | 0/0 | 348 | 0.0000 |
| binary_only | frame_uniform | +6.0386 [+2.5079,+10.2195] | +2.6232 [-1.5751,+6.4438] | 0/348 | 150/160 | 183 | 306.2820 |
| binary_only | independent | +10.0020 [+4.4124,+15.9594] | +5.3419 [-2.5947,+12.1291] | 7/348 | 283/277 | 55 | 428.9130 |
| binary_only | joint_free | +11.2697 [+4.9869,+17.8645] | +4.8097 [-3.0442,+11.6625] | 6/348 | 279/281 | 55 | 428.9130 |
| binary_only | joint_matched | +10.8402 [+4.8043,+17.2178] | +4.6016 [-3.0429,+11.3151] | 6/348 | 281/279 | 56 | 428.9130 |
| binary_only | uncontrolled_neural | +16.4074 [+7.4982,+25.5310] | -5.0187 [-16.4749,+5.0800] | 6/348 | 304/320 | 15 | 1742.9603 |
| legacy_binary_cached | floor | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] | 0/348 | 0/0 | 348 | 0.0000 |
| legacy_binary_cached | frame_uniform | +5.2071 [+2.2020,+9.1758] | +2.0541 [-1.9341,+5.8289] | 0/348 | 143/151 | 194 | 277.9125 |
| legacy_binary_cached | independent | +9.7080 [+4.2118,+15.5267] | +5.1618 [-2.9968,+12.0370] | 7/348 | 285/280 | 55 | 309.5896 |
| legacy_binary_cached | joint_free | +11.3326 [+5.0080,+17.9928] | +5.1034 [-3.3339,+12.3919] | 7/348 | 281/283 | 52 | 340.5715 |
| legacy_binary_cached | joint_matched | +10.7543 [+4.6989,+17.1264] | +4.7033 [-3.3526,+11.6839] | 4/348 | 283/282 | 55 | 340.5715 |
| legacy_binary_cached | uncontrolled_neural | +16.4074 [+7.4982,+25.5310] | -5.0187 [-16.4749,+5.0800] | 6/348 | 304/320 | 15 | 1742.9603 |

The legacy arm is cached_verified, not freshly inferred here or input-
matched to the new seven-amplitude inventory. The two new heads share
the same inputs/capacity; their treatment changes active supervision and
action availability. The shared-weight binary deployment isolates only
the secondary inference effect. All use the same frozen physical floor.
Original CV defines easy labels; reference moments are not alpha-scaled.
Actual candidate-to-floor envelopes are shared by both new heads.
The floor has no interventions and therefore no finite selected-risk
support; this does not mean it causes harm. Its zero path change is
analytical equality, not an empirical finite-support risk certificate.

## Prespecified Subsets And Error Diagnostics

| Model, joint-free | Easy known gain (%) [95% CI] | Hard known gain (%) [95% CI] | Available-label ADE (pixel) [95% CI] | Group p95 ADE (pixel) [95% CI] | Available-endpoint FDE diagnostic (pixel) [95% CI] |
|---|---:|---:|---:|---:|---:|
| amplitude_bank | not estimable | +14.8608 [+8.7763,+21.2597] | +57.5001 [+35.6225,+82.8318] | +160.6129 [+96.8973,+238.7126] | +147.2992 [+90.1738,+217.3593] |
| amplitude_bank_binary_deploy | not estimable | +14.3892 [+8.3418,+20.8340] | +57.9709 [+35.8864,+83.5308] | +162.8997 [+97.9340,+243.0974] | +148.5679 [+90.7054,+219.7602] |
| binary_only | not estimable | +14.0786 [+8.3472,+20.2668] | +57.7213 [+35.7420,+83.0947] | +162.4746 [+97.5507,+242.5372] | +148.0391 [+90.5842,+218.5360] |
| legacy_binary_cached | not estimable | +14.2552 [+8.4302,+20.5928] | +57.6756 [+35.7162,+83.0830] | +162.3952 [+97.5477,+242.3817] | +147.6542 [+90.3187,+218.1416] |

These are source-macro secondary diagnostics after producer/seed
averaging, not a new primary endpoint. Positive subset gain means less
error; it cannot replace the worst-recording easy-risk gate. The tail
column averages within-group p95 values, not a pooled population p95.
The endpoint diagnostic uses only observed final endpoints; full-roster
FDE remains not estimable when any required view lacks full endpoints.
A missing required subset stays not estimable, never dropped. All six
policies and the full-roster FDE status are in `secondary_metrics.json`.

## Intervention Size And Matched Utility

| Model | Source-macro switch (%) | Source-macro mean amplitude, all rows | Empty matched group views |
|---|---:|---:|---:|
| binary_only | 63.2371 | 0.632371 | 4/138 |
| amplitude_bank | 70.9258 | 0.658174 | 3/138 |
| amplitude_bank_binary_deploy | 66.1899 | 0.661899 | 4/138 |

Means average within source group before averaging groups. Raw amplitude
histograms retain all repeated row views in the machine-readable receipts.
Equal intervention counts do not mean equal amplitude/path-change mass.
Same-count controls measure utility only; empty comparisons are not
estimable, not safe zeros. Full-policy risk uses every required recording.

## Fitting-Side Reliability

These are pooled known-cost mass ratios over all 18 fitting jobs, not
independent or official macro scores. Fixed regions include candidate
identity, not just selected agents. All are evaluated before new held
readout; no region or checkpoint is selected from those outcomes.

| Scoring head | Fixed causal action region | H / predicted H | EH / predicted EH |
|---|---|---:|---:|
| binary_only | all_legal_full_amplitude | 1.5796 | 0.6398 |
| binary_only | binary_only_independent | 103.4962 | 731.8533 |
| binary_only | amplitude_bank_independent | 14.7061 | 13.9463 |
| binary_only | amplitude_bank_binary_independent | 26.8975 | 24.4401 |
| amplitude_bank | all_legal_full_amplitude | 1.5362 | 0.7035 |
| amplitude_bank | binary_only_independent | 21.1430 | 19.0989 |
| amplitude_bank | amplitude_bank_independent | 42.7012 | 202.2497 |
| amplitude_bank | amplitude_bank_binary_independent | 91.8027 | 768.9493 |

## Solver Completion And Fallback

| Deployment arm | Repeated query views | Nonoptimal matched solves | Nonoptimal free solves | Free choices changed from independent |
|---|---:|---:|---:|---:|
| binary_only | 16734 | 19 | 37 | 10126 |
| amplitude_bank | 16734 | 323 | 175 | 11891 |
| amplitude_bank_binary_deploy | 16734 | 23 | 45 | 11139 |

These queries repeat sources/producer/seed contexts. Timeout or failed
post-check retains the independent parent feasible under predicted
budgets; this is not an empirical safety pass or proof of an exact
joint optimum. Complete reason counts remain in `solver_summary.json`.

## Execution And Limits

Training/refits: 3025.523 s, peak RSS 8,712,224,768 bytes.
Held evaluation: 910.433 s; separate replay: 199.648 s.
Native arm64 CPU4/interop1/workers0, atomic checkpoints and exact resume.
Fitting-loss panels use different active-action targets; smaller values
in one panel do not by themselves show better learning than the other.
Lines are medians and shading is the min-max over 18 fitting runs,
not validation confidence intervals or independent experimental units.
Saved solver bindings/feasibility replay is not bitwise MILP reoptimization.
Joint allocation pools predicted budgets, not geometric collision costs.
Required views repeat producer/seed contexts, not independent recordings.
No unsupported or empty-event view is dropped to manufacture a risk pass.
Available-label ADE and bounds for wholly unknown futures do not identify
unobserved steps within partial trajectories. Pixels/native 8/12 steps
are not verified metres/seconds or SDD raw t50. No new trajectory backbone,
JEPA lift, independent confirmation, deployment, physical-safety, true3D,
foundation or submission-ready claim. Stage5C execution and SMC stay off.
