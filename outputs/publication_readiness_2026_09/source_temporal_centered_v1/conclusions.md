# Temporal-Centering Repair: Completed Comparison

## Conclusion

Both fixed repairs still lose to stationary CV; no deployable forecasting benefit is established.

All 24 new heads completed 10,000 updates each. There was no held-based arm, seed,
checkpoint, threshold or cohort selection. The 36 original matched control heads
are `cached_verified` and rescored, not newly trained.

## Scope

Same 15,430 stationary-history queries, 29 recordings, four explored source sites,
seeds 17/29/43. Eight observed and twelve future annotation steps at stride 12 raw
frames. Not the main benchmark, raw-frame t+50 supplement or a Stage37 rerun.
Pixel/local annotations only, no metric or seconds equivalence. Supplied
histories may use later interpolation controls: offline, not sensor-as-of.

Centering removes each observation window's mean frozen appearance. The unit
variant also divides by observed RMS variation with a fixed 0.001 floor. Both keep
geometry, coverage, loss, 63,960 parameters, initialization, sampler and update budget.
ResNet18 remains frozen; this is not end-to-end encoder training.

## Actual Forecasts And Paired Comparisons

| Arm | Source | Equal-site ADE gain (%) | Conditional site CI | Easy harm (annotation px) | Positive held fits |
| --- | --- | ---: | --- | ---: | ---: |
| geometry | cached_verified_recomputed | -0.070 | [-0.164, -0.014] | 0.001644 | 0/12 |
| current | cached_verified_recomputed | -1.908 | [-2.942, -1.020] | 0.041487 | 0/12 |
| sequence | cached_verified_recomputed | -6.102 | [-9.171, -3.807] | 0.120212 | 0/12 |
| centered | fresh_run | -0.762 | [-1.743, -0.123] | 0.020320 | 0/12 |
| centered_unit | fresh_run | -1.756 | [-3.425, -0.554] | 0.041864 | 0/12 |

| Fixed contrast | Difference (pp) | Conditional CI |
| --- | ---: | --- |
| centered minus sequence | +5.340 | [+3.667, +7.578] |
| centered_unit minus sequence | +4.346 | [+3.210, +5.864] |
| centered minus geometry | -0.692 | [-1.582, -0.104] |
| centered_unit minus geometry | -1.686 | [-3.264, -0.535] |
| centered_unit minus centered | -0.994 | [-1.714, -0.431] |

Primary is ratio of equal-site mean normalized errors, not mean site percentages.
Seeds average errors, not predictions. The 2,000 shared physical-site bootstrap draws
are conditional on four explored sites and overlapping fitting populations.
No independent confirmation or multiplicity-adjusted guarantee is claimed.
An improved contrast against a failing model is not itself positive forecasting.

## Every Site And Seed

| Arm | Site | Train gain (%) | Held gain (%) | Hard gain (%) | Seed held gains (%) |
| --- | --- | ---: | ---: | ---: | --- |
| geometry | coupa | -0.013 | -0.015 | -0.001 | -0.014, -0.016, -0.016 |
| geometry | deathCircle | -0.013 | -0.012 | -0.000 | -0.011, -0.013, -0.011 |
| geometry | gates | -0.010 | -0.037 | +0.000 | -0.038, -0.031, -0.041 |
| geometry | hyang | -0.013 | -0.217 | -0.015 | -0.273, -0.164, -0.215 |
| current | coupa | +0.015 | -0.747 | +0.001 | -0.806, -0.566, -0.868 |
| current | deathCircle | +0.121 | -1.499 | -0.011 | -1.425, -1.498, -1.573 |
| current | gates | +0.034 | -2.365 | -0.052 | -2.281, -2.255, -2.561 |
| current | hyang | +0.353 | -3.316 | +0.156 | -3.283, -3.281, -3.384 |
| sequence | coupa | +0.188 | -3.153 | -0.027 | -3.502, -2.974, -2.982 |
| sequence | deathCircle | +0.550 | -4.422 | -0.030 | -4.763, -3.918, -4.585 |
| sequence | gates | +0.268 | -8.414 | -0.209 | -9.559, -8.026, -7.655 |
| sequence | hyang | +1.499 | -9.662 | +0.608 | -10.239, -9.064, -9.683 |
| centered | coupa | -0.014 | -0.052 | -0.001 | -0.047, -0.037, -0.072 |
| centered | deathCircle | -0.008 | -0.224 | +0.002 | -0.173, -0.208, -0.293 |
| centered | gates | -0.014 | -0.475 | +0.018 | -0.628, -0.274, -0.524 |
| centered | hyang | +0.237 | -2.318 | +0.010 | -2.132, -2.382, -2.440 |
| centered_unit | coupa | -0.007 | -0.339 | -0.003 | -0.287, -0.304, -0.426 |
| centered_unit | deathCircle | +0.070 | -0.839 | +0.009 | -0.671, -0.874, -0.973 |
| centered_unit | gates | +0.008 | -1.614 | +0.053 | -1.880, -0.917, -2.045 |
| centered_unit | hyang | +0.547 | -4.404 | +0.062 | -4.383, -4.080, -4.750 |

## Magnitude And Oracle

| Arm | Native ADE | Native FDE | Nonzero-target gain (%) | ADE p95 | ADE p99 | Binary oracle (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| geometry | 1.128781 | 2.365927 | -0.019 | 4.081840 | 17.570326 | 0.007628 |
| current | 1.151275 | 2.472454 | -0.052 | 4.084130 | 17.566243 | 0.400328 |
| sequence | 1.201022 | 2.606171 | -0.588 | 4.108779 | 17.568691 | 1.056320 |
| centered | 1.139625 | 2.407879 | -0.061 | 4.083270 | 17.570309 | 0.163048 |
| centered_unit | 1.152787 | 2.450537 | -0.168 | 4.073393 | 17.570505 | 0.336776 |

Native errors are annotation-pixel descriptions, not cross-dataset metric means.
Zero-target CV error is zero; percentage easy degradation is undefined, not a
2% pass. Hard cutoffs use training-complement labels. Future-oracle minima are
label-only diagnostics, not a deployable switch rule.

## Verification And Compute

Fresh training totals 240,000 updates and 764.240s summed fitting, including
the 100-update pilot. Native arm64, CPU 4/inter-op 1/workers 0. All 24 train/held forecasts
replay exactly. All 24 sample streams match cached controls; all 15,430 real input
transforms pass shared-offset and batch-composition checks. Six OOF archives
recompute, and completed resume adds zero updates while preserving
83 artifacts. Frozen image features and upstream
provenance are checked through the parent manifest. No raw data or checkpoints in Git.

Fresh work: input audit, 24 heads, analysis and verification. `cached_verified`:
image embeddings and 36 control heads. `not_run`: main/outer forecasts, new policy,
independent confirmation and external evaluation. No new deployment, Stage5C or SMC.
Not submission-ready. See [registered design](../source_temporal_centered_decision.md),
[analysis](analysis.json), [verification](verification.json),
[failure analysis](failure_analysis.md) and [evidence gates](gates.md).
