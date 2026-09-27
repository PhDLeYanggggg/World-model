# Source Observation Quality and Neighbor Coverage

## Material Passport
Fresh raw-source audit; cached inputs hash-verified. Source development only.
No model training, future outcome scoring or independent confirmation.

## Exact Source Checks
318,969 complete-history target queries from 163 recordings and 12 source localities.
All original packed geometry and 2,551,752 observed boxes match exactly.
The raw archive checksum and each admitted member row hash are checked.
Whole raw files are parsed for identity checks; per-query diagnostics use only
[query-84, query]. Future label arrays and reserved recordings are not opened.

## Measured Omission
282,529/318,969 queries (88.58%) have a partial-history neighbor in the nearest eight currently visible agents.
The new neighbor geometry changes on 282,529 queries (88.58%).
Mean neighbor count: 6.7624 legacy, 7.3045 masked.
The old source cache contains these visible agents, but the old model packer
excludes them because target eligibility was also used for neighbor eligibility.
This is a demonstrated representation omission, not proof of useful interaction lift.

## Per-Locality Diagnostics
| Locality | Queries | Partial neighbor % | Raw-prefix frame presence | Line RMS / width median | FD prefix error | OLS4 prefix error | OLS6 prefix error |
|---|---:|---:|---:|---:|---:|---:|---:|
| eu-locality-007 | 5664 | 77.86 | 0.9879 | 0.0751 | 0.1728 | 0.1724 | 0.1903 |
| eu-locality-008 | 161653 | 92.76 | 0.9922 | 0.0499 | 0.1613 | 0.1523 | 0.1657 |
| eu-locality-020 | 1282 | 30.19 | 0.9745 | 0.0531 | 0.1357 | 0.1100 | 0.1177 |
| eu-locality-048 | 8344 | 95.07 | 0.9923 | 0.0922 | 0.1992 | 0.2041 | 0.2403 |
| eu-locality-067 | 7105 | 87.85 | 0.9876 | 0.0997 | 0.2240 | 0.2206 | 0.2520 |
| eu-locality-074 | 104353 | 81.88 | 0.9921 | 0.0790 | 0.2074 | 0.2082 | 0.2302 |
| eu-locality-082 | 1448 | 66.02 | 0.9878 | 0.0576 | 0.1807 | 0.1813 | 0.2093 |
| eu-locality-110 | 8660 | 91.13 | 0.9918 | 0.0413 | 0.1696 | 0.1718 | 0.1859 |
| eu-locality-112 | 5131 | 98.30 | 0.9807 | 0.0672 | 0.2147 | 0.1951 | 0.2067 |
| eu-locality-119 | 5787 | 94.83 | 0.9892 | 0.0855 | 0.2053 | 0.2181 | 0.2541 |
| eu-locality-124 | 3651 | 86.09 | 0.9894 | 0.0721 | 0.3000 | 0.3350 | 0.3761 |
| eu-locality-126 | 5891 | 95.82 | 0.9918 | 0.0358 | 0.1493 | 0.1561 | 0.1788 |

## Motion and Detector Proxies
| Diagnostic | Mean | Median | P90 | P99 |
|---|---:|---:|---:|---:|
| path_over_width | 2.0921 | 0.830816 | 5.74321 | 10.7533 |
| line_residual_over_width | 0.104112 | 0.0638552 | 0.228619 | 0.680869 |
| last_fd_ols8_disagreement_over_width | 0.126966 | 0.0666155 | 0.295833 | 0.907439 |
| width_range_over_width | 0.215541 | 0.179105 | 0.420344 | 0.777711 |
| height_range_over_height | 0.129527 | 0.101164 | 0.251174 | 0.580584 |
| half_pixel_fraction | 4.38914e-05 | 0 | 0 | 0 |
| stationary_step_fraction | 0.000558048 | 0 | 0 | 0 |
| reversal_fraction | 0.323667 | 0.333333 | 0.666667 | 1 |
| zero_current_width | 0 | 0 | 0 | 0 |
| observed_prefix_fd_error | 0.18222 | 0.115916 | 0.397214 | 1.10939 |
| observed_prefix_ols4_error | 0.178331 | 0.0976936 | 0.403839 | 1.25424 |
| observed_prefix_ols6_error | 0.196606 | 0.0995949 | 0.458663 | 1.44473 |
| neighbor_geometry_changed | 0.885757 | 1 | 1 | 1 |
| partial_nearest_neighbors | 2.51436 | 2 | 5 | 7 |
| has_partial_nearest_neighbor | 0.885757 | 1 | 1 | 1 |
| legacy_neighbor_count | 6.76239 | 8 | 8 | 8 |
| repaired_neighbor_count | 7.30448 | 8 | 8 | 8 |
| repaired_missing_slots | 13.4872 | 9 | 33 | 64 |
| raw_prefix_frame_presence | 0.991605 | 1 | 1 | 1 |
| raw_prefix_max_gap | 1.50198 | 1 | 3 | 8 |
| raw_prefix_mean_detector_confidence | 0.740642 | 0.783605 | 0.88103 | 0.910638 |

## Interpretation Boundaries
Fixed prefix forecasts use observed steps 1-6 to predict observed steps 7-8.
Errors are in current-box-width units with a one-coordinate-unit denominator floor.
They are not future ADE/FDE, learned estimator gains, or metric/seconds claims.
Each locality is shown, without choosing a favorable one. Windows overlap;
these distribution summaries are descriptive, not independent confidence intervals.
Box motion, reversal and line residuals may reflect real dynamics or detector noise.
High raw frame presence does not prove identity correctness or annotation accuracy.
Sparse event-label attribution was not rerun; no new label-driven filter was selected.

## Repair and Next Test
The versioned input repair retains all current-visible neighbor candidates,
selects the nearest eight by current position, and zeros masked coordinates/times.
Ego history, baseline rollout, target rows and raw future labels are unchanged.
Do not deploy old weights on the new schema as if retraining had happened.
A separate neural adapter also replaces complete-history attention eligibility
and includes valid partial tokens in past-only conditioning. It retains the
original parameter budget, zero-initial correction and motion-bounded output.
Synthetic forward/backward checks are code tests, not trained model evidence.
Next: one matched legacy-vs-masked-neighbor retraining experiment with the
same full producer-chain exclusions, fixed training budget and three seeds.
This isolates an observed input defect; it does not assume smoothing or more
context works. Keep cap, losses, target rules and policy frozen for that contrast.
Independent selection/calibration/confirmation stay closed. Stage5C and SMC stay off.
