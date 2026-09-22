# Temporal Intervention Improves Easy Protection, Not the Primary Gain

## Evidence Status

2026-09-22. **fresh_run**: 24 real Torch scalar cost-head fits, fixed readout,
checkpoint replay, separate arithmetic checks and frozen-choice diagnosis.
**cached_verified**: nested source-excluded neural forecasts, causal features,
the original scalar-log reference and past scene context. **not_run**: new full
trajectory-predictor training, independent calibration/confirmation, external
readout and the raw-frame t+50 supplement.

Registration `d5f6d12f` preceded training. No threshold, schedule, seed, metric or
data role changed after readout. The task is eight observed to twelve predicted
annotation steps at SDD raw stride 12, in annotation pixels. All four physical
sites have informed development. This is not a metric, seconds-level, true-3D,
foundation-model or independent final-test result. Stage5C and SMC are off.
No model is promoted or deployed.

## Matched Experiment

The previous all-prefix guard rejected almost every switch at the first future
step. This experiment changes the action rather than weakening the threshold:
keep the first forecast point at causal constant velocity (CV), then increase
the neural contribution linearly to the final point. A uniform blend is matched
to the same mean displacement from CV **for every query**. Its coefficient uses
forecasts, never labels or future availability.

Both arms receive new scalar benefit/harm heads with the same source exclusions,
complete fitting support, fixed emphasis, sampler draws, three seeds, optimizer
and 12,000-update budget. Candidate-derived features and labels are recomputed,
including train-only normalization under the same rule. The strict rule remains
predicted harm <= 0.1 predicted benefit, with the past-stop veto. Each head has
45,954 parameters and 356 inputs. These cost regressors are not calibrated
failure probabilities.

All 24 fits completed: 288,000 updates, 73,728,000 draws, zero unknown-label draws,
and 241.35955 recorded fitting seconds. This excludes preparation, hashing,
evaluation and upstream forecast training. A real 100-update pilot was resumed.
Private checkpoints retain loss traces and optimizer/sampler states. Finite loss
does not establish downstream success.

## Fixed Results

These are equal-physical-site percentage improvements over CV, averaging seed
errors rather than forecasts. Negative easy degradation means improvement.
Switch counts sum repeated seed instances, not independent trajectories.

| Fixed policy | ADE gain | FDE gain | Hard gain | Aggregate easy degradation | Worst scene/seed easy degradation | Switches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Temporal ramp, strict: primary** | **3.39756%** | **4.90029%** | **3.67549%** | **-1.24935%** | **0.91149%** | **33,793** |
| Matched uniform, strict: control | 3.64449% | 3.85268% | 3.95700% | -0.63522% | 2.56229% | 36,105 |
| Uniform ranking at ramp switch count | 4.54016% | 4.88186% | 6.83724% | 3.66766% | 6.29827% | 33,793 |
| Ramp at uniform choices | 3.51501% | 5.05218% | 3.76532% | -1.34917% | 0.77868% | 36,105 |
| Uniform at ramp choices | 3.52230% | 3.72569% | 3.85960% | -0.52653% | 2.77878% | 33,793 |
| Ramp without learned risk rejection | 10.19683% | 13.10757% | 11.44447% | 10.56207% | 18.17490% | 399,141 |
| Uniform without learned risk rejection | 10.91615% | 11.46770% | 11.87898% | 12.10561% | 20.92503% | 399,141 |

The primary difference is **-0.24693 percentage points**, paired 3,000-physical-site
bootstrap CI **[-0.29914, -0.19472] pp**. Ramp loses to uniform in all four sites.
Against the verified original-forecast scalar-log reference (4.18873% ADE gain,
itself unsafe), it loses **0.79117 pp**, CI **[-1.23722, -0.45840] pp**. At equal
switch count it loses **1.14260 pp**, CI [-1.67709, -0.64329] pp.

Ramp versus CV has CI [2.29647%, 4.86945%]. That positive contrast does not rescue
the failed stronger comparisons. FDE improves more than uniform, but FDE cannot
replace the primary ADE rule after readout. No secondary winner is promoted.

| Primary ramp, seed-averaged site | ADE gain | Easy degradation |
| --- | ---: | ---: |
| coupa | 3.35282% | -4.93520% |
| deathCircle | 5.64448% | 0.21419% |
| gates | 2.04858% | 0.38750% |
| hyang | 2.54436% | -0.66389% |

The positive-error easy ceiling passes in every scene/seed. However, **one
complete exact-zero-CV case is harmed**, in hyang seed17: ADE rises from 0 to
**3.90669 annotation pixels**, not numerical dust. Both strict arms switch on
that same case; predicted harm is 0.18957 for ramp and 0.15362 for uniform.
This violates the separate exact-zero guard. Do not hide it in the easy average
or add an epsilon to convert zero-reference error to a percentage.

## Failure Diagnosis

The additional diagnosis is post-readout and changes no decisions:

1. **Action shape trades accuracy for protection.** At identical ramp choices,
   ramp loses 0.12474 pp ADE gain versus uniform but reduces worst scene/seed
   easy degradation from 2.77878% to 0.91149%. At identical uniform choices,
   it loses 0.12948 pp but reduces worst easy degradation from 2.56229% to
   0.77868%. Protection is not explained solely by switching fewer rows.
2. **The decisions also lose useful gain.** Using uniform actions with the two
   choice sets gives another -0.12219 pp contribution. Added to -0.12474 pp,
   this exactly reconstructs the -0.24693 pp primary gap. The reverse crossover
   path agrees. This is fixed-action arithmetic, not an independent causal estimate.
3. **Risk remains underestimated.** Both heads underestimate realized mean harm
   on selected complete-label rows in all twelve source/seed views. Smaller or
   smoother interventions do not make their expected-risk estimates calibrated.
4. **Early protection removes useful early correction.** Ramp prefix-ADE gain
   is 0% at point one by construction, 0.2805% over two points and 0.9697% over
   four. Uniform strict gives 0.6826%, 1.8679% and 2.8592%. This is consistent
   with reduced average-path gain despite a better endpoint; it does not prove
   that every possible temporal schedule fails.
5. **Equal count is not equal movement budget.** Per-query displacement is
   matched, but different selected rows have different displacement. Summed
   selected displacement is 596,783.99 for ramp strict, 619,970.75 for uniform
   strict and 1,012,632.28 for equal-count uniform ranking, across repeated seeds.
   Identical-choice controls match this quantity exactly.

## Path and Interaction Proxies

The native discrete second-difference proxy is lower for ramp at identical choice
sets in every site. Strict-policy site means, ramp/uniform: coupa 0.08654/0.12476,
deathCircle 0.17063/0.20264, gates 0.08839/0.11758 and hyang 0.06132/0.08184.
CV is almost straight by construction. These are pixel-space discrete smoothness
proxies, not physical acceleration or evidence that every turn is undesirable.

Past-context proximity is evaluated on 20,932 scene queries per seed, with 265,016
supported edges, 7,299 unsupported edges and 5,748 unsupported context occurrences.
Ramp strict has lower total excess proxy cost than uniform strict in each seed:
18.4518 vs 20.9030, 17.7104 vs 20.0662, and 13.7444 vs 18.2092. It is not better
in every site/seed. Choice sets differ; the JSON preserves crossover diagnostics.
No safety is imputed to unsupported edges. No new joint optimization or physical
safety certificate was trained or established.

## Support and Verification

There are 175,756 past-eligible queries: 172,957 with ADE labels, 144,010 with
endpoint labels and 143,918 with complete twelve-step labels. Ramp selects
6.4091% of indexed query/seed instances. Of 33,793 switches, 5,210 have incomplete
future labels and 575 have none. Unknown cases are not counted as harmless.
Full-grid gain lower bounds remain negative in gates for all three seeds.
Future support is never used to choose an inference action.

All 24 checkpoints replay, covering 1,054,536 cost-score rows. Separate formulas
check 3,163,608 reused fitting-arm row instances, 84 policy choices and 672 scene
reductions, transformed labels, draw matching and missing-label bounds. Prefix
and interaction diagnostics are checkpoint-replayed, not separately reimplemented.
The new diagnosis separately checks gate arithmetic and reproduces exactly.
Eight new diagnostic tests pass; the 21 unchanged scoped tests remain valid.
The full legacy suite was not rerun.

Checks share source arrays and the same implementing agent: engineering
verification, not independent research confirmation. Three seeds and 3,000
resamples of four development-exposed sites do not create independent calibration
data. The dataset-provided annotation contract is not a verified live-sensor
annotation-provenance claim; self-audited priors are not human gold.

## Decision and Next Work

**Do not deploy this repair.** Primary superiority, old-reference superiority
and exact-zero conditions fail despite improved positive-error easy protection.
Historical Stage26/37 are not recertified. The pinned manuscript stays unchanged;
this is a separate negative follow-up, **not yet a CVPR submission candidate**.

The highest research priority remains conditional risk on the rows the model
admits, with genuinely separate calibration support. On existing development
assets, compare fitting versus held-source conditional errors and exact-zero
moving-row support before registering another targeted repair. Do not sweep
schedules, thresholds or exclusions on these outcomes, or promote secondary FDE.
Independent-source admission and calibration-role decisions remain pending;
meaningful joint-agent evidence and external confirmation are still missing.

See [execution and reproduction](execution_notes.md), [fixed protocol](registration.md),
[aggregate metrics](analysis.json) and [action/choice diagnosis](action_choice_diagnosis.json).
