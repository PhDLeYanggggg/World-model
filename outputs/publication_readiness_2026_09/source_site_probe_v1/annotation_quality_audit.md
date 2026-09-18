# Annotation-Change and Past-Context Audit

Descriptive follow-up; no labels, row eligibility, training or thresholds changed.
The positive label is any future annotation-coordinate change, not a verified intention event.
Displacements below are restored to annotation pixels, never meters or seconds.

| Site | Positive rows | Median maximum future pixel displacement | Median fraction of current box diagonal | Positive change <=1px | Positive change <0.1 box diagonal | Any occluded past frame | Any generated past frame | Any past neighbor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_source | 10039 | 2.061553 | 0.034602 | 2248 | 8583 | 1760 | 22374 | 22323 |
| bookstore | 3175 | 2.000000 | 0.034655 | 949 | 2757 | 426 | 6944 | 6944 |
| coupa | 2546 | 1.802776 | 0.028146 | 435 | 2341 | 215 | 4250 | 4237 |
| deathCircle | 987 | 4.000000 | 0.046653 | 154 | 710 | 674 | 3054 | 3054 |
| gates | 420 | 2.236068 | 0.045077 | 76 | 328 | 292 | 1662 | 1662 |
| hyang | 2911 | 2.236068 | 0.036418 | 634 | 2447 | 153 | 6464 | 6426 |

All22,374 supervised source histories have exact eight-frame/agent past joins at stride12.
Generated/occluded flags come from dataset annotation metadata; absence of a generated flag is not a guarantee of online sensor provenance.
Nonzero coordinate change can include rounding/interpolation/body-box change; this audit does not establish which mechanism caused an event.
The one-pixel bin allows1e-6annotation-pixel roundoff after float32 target restoration; this does not change the binary training label.
The current-box diagonal differs from the historical median-past-box denominator, even when aggregate counts agree. Do not silently exchange definitions.
Neighbor presence does not prove interaction intent. Main box labels are not opened or guessed.
No main protocol change, sealed-role access, new deployment, Stage5C or SMC.

## Label Frequency Versus Trajectory Error Mass

Supplementary source-only diagnosis. Exact stationary histories give a zero-displacement CV rollout.
The table partitions source CV ADE error mass, not neural prediction performance or the main primary endpoint.
| Source displacement slice | Windows | Fraction of all stationary windows | Fraction of stationary CV ADE error mass |
| --- | ---: | ---: | ---: |
| unchanged | 12335 | 55.131% | 0.000% |
| positive_below_0.1_box | 8583 | 38.361% | 43.406% |
| 0.1_to_below_0.5_box | 1212 | 5.417% | 30.439% |
| at_least_0.5_box | 244 | 1.091% | 26.155% |

Window-pooled source diagnostic only; per-site values are retained in the JSON.
These descriptive slices are not new training labels, sampling weights, deployment thresholds or a replacement metric.
