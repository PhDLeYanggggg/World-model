# Full-Range Bounded Harm: Complete Matched Experiment

## Material Passport

Fresh18 fits,36,000 updates in185.18s;144 readout views in112.77s.
Six old controls are cached_verified models with fresh exactly matching
inference. No old trajectory predictor or new data role was trained/opened.
This is not independent confirmation.
The18 new finals were all frozen before this readout. No checkpoint or
threshold selection. Native arm64 CPU4/workers0, read-only CREATE TRAIN
streaming; no new scheduler job.

## Controlled Change

The harm branch uses additive probability shifts across its full causal
range instead of bounded log-odds shifts. Benefit/reference outputs remain
exactly fixed. Initial weights, source/query sampling, draw hashes, optimizer
settings, budget and12,258 trainable/25,407 total parameters match the
isolated-odds control. Range and gradient geometry both change; they are
not separately identified. This is a cost head, not a dynamics forecaster.

Utility continuation, joint risk and isolated odds have the same4,000
effective head updates including their2,000-step anchors. Local, anchor
and old-risk controls have2,000 updates and are not equal-total-budget
controls. No favorable comparator is omitted.

## Complete Results

### Fit

| Policy | Signed MSE | Selected occurrences | Known easy violations /72 | Upper easy violations /72 | Empty | Undefined easy | Full support /72 |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 0.559739 | 123490 | 1 | 59 | 1 | 7 | 5 |
| anchor | 0.489310 | 337984 | 62 | 65 | 4 | 6 | 1 |
| old_risk | 1.965634 | 67774 | 9 | 57 | 3 | 7 | 7 |
| utility_continue | 0.396041 | 401255 | 65 | 66 | 1 | 6 | 0 |
| joint_risk | 1.224959 | 76621 | 6 | 62 | 3 | 6 | 4 |
| isolated_risk | 1.985264 | 45395 | 21 | 54 | 5 | 6 | 11 |
| full_range | 2.258883 | 58707 | 25 | 66 | 0 | 6 | 0 |

| Full-range minus control | Full lower utility pp [95% CI] | Matched-count lower utility pp [CI] | Signed MSE difference [CI] |
|---|---:|---:|---:|
| local | -2.301961 [-4.106424, -0.668867] | -0.505265 [-1.114489, -0.074146] | 1.699144 [0.761879, 2.961123] |
| anchor | -1.386810 [-3.384938, 0.429974] | 0.394905 [-0.300243, 1.388070] | 1.769573 [0.868652, 2.989154] |
| old_risk | 0.762784 [-1.188234, 2.595397] | 0.455908 [0.054521, 0.898250] | 0.293249 [-0.444648, 1.314192] |
| utility_continue | -3.583291 [-5.776375, -1.637711] | -0.436994 [-0.885221, -0.053560] | 1.862842 [0.966470, 3.088786] |
| joint_risk | 0.241275 [-0.943148, 1.518830] | 0.127356 [-0.055748, 0.323015] | 1.033924 [0.383224, 2.094242] |
| isolated_risk | 1.661293 [-0.153880, 3.570063] | 0.487896 [0.122105, 0.951194] | 0.273619 [-0.300437, 0.930665] |

### Held

| Policy | Signed MSE | Selected occurrences | Known easy violations /72 | Upper easy violations /72 | Empty | Undefined easy | Full support /72 |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 4.013844 | 57903 | 46 | 65 | 3 | 7 | 0 |
| anchor | 1.695168 | 154772 | 61 | 71 | 0 | 1 | 0 |
| old_risk | 7.452018 | 27826 | 40 | 56 | 5 | 12 | 1 |
| utility_continue | 1.967093 | 182163 | 61 | 70 | 0 | 2 | 0 |
| joint_risk | 4.663019 | 36085 | 45 | 60 | 4 | 9 | 0 |
| isolated_risk | 5.047722 | 24204 | 47 | 63 | 3 | 4 | 3 |
| full_range | 7.774976 | 31270 | 44 | 59 | 4 | 12 | 0 |

| Full-range minus control | Full lower utility pp [95% CI] | Matched-count lower utility pp [CI] | Signed MSE difference [CI] |
|---|---:|---:|---:|
| local | -1.846424 [-3.847599, -0.246056] | -0.038611 [-0.394379, 0.345017] | 3.761132 [0.423190, 7.649969] |
| anchor | -2.365346 [-3.797122, -1.090739] | 0.074553 [-0.526357, 0.746414] | 6.079808 [2.827737, 9.664212] |
| old_risk | -0.237884 [-1.927101, 1.240055] | 0.102056 [-0.111769, 0.349878] | 0.322958 [-2.335677, 2.689552] |
| utility_continue | -3.111151 [-4.398100, -1.910054] | -0.261395 [-0.696439, 0.147135] | 5.807884 [2.727058, 9.230691] |
| joint_risk | -0.846565 [-2.100371, 0.172569] | 0.006613 [-0.290089, 0.314868] | 3.111958 [1.143061, 5.277942] |
| isolated_risk | -0.162919 [-1.830483, 1.431466] | 0.243378 [-0.057308, 0.590769] | 2.727254 [0.628834, 5.194862] |

## Registered Gate

Advance to later development design: **False**.
Failures: local_full_not_positive, local_matched_not_positive, anchor_full_not_positive, anchor_matched_not_positive, old_risk_full_not_positive, old_risk_matched_not_positive, utility_continue_full_not_positive, utility_continue_matched_not_positive, joint_risk_full_not_positive, joint_risk_matched_not_positive, isolated_risk_full_not_positive, isolated_risk_matched_not_positive, full_policy_support_failed.

Positive lower95% intervals required against every control at both full
and matched counts, plus full-policy risk support in all72 held views.
Empty/undefined or incomplete-label policies do not earn a safety pass.
A lower loss/MSE or one positive source cannot substitute for this rule.

Utility is pair-specific completion-lower mass divided by full-known
reference mass, NOT ADE/FDE. Bounds are not transitive. Original TRAIN
fit/held recordings are already exposed. Bootstrap3,000 averages repeated
contexts/seeds within12 localities; nominal, dependent and not multiplicity
adjusted or independent confirmation. Three cost seeds are not three
new forecasting producers. No deployable world-model success is inferred.

## Reproduction

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_full_range_harm check
.venv-pytorch/bin/python -m scripts.report_m3w_full_range_harm --check
```

Complete outputs are immutable; only unfinished phases permit --resume.
Private checkpoints and raw histories are excluded from Git. Public outputs
contain aggregate metrics, provenance and reproducible code.

Obs8/pred12 raw stride12, detector-track pixel/local coordinates. No metric,
seconds, physical safety, human gold, true3D or foundation claim.
Stage5C/SMC disabled. Historical Stage37 is exploratory, not recertified.
