# Development Error Concentration

Fresh descriptive analysis of hash-verified caches; no metric, threshold or sample selection changes.

| Seed | Recording | Rows | Top ~1% rows | Normalized floor error share % | Native floor error share % |
| --- | --- | ---: | ---: | ---: | ---: |
| 17 | ucy_students01 | 891 | 8 | 24.114 | 2.191 |
| 17 | ucy_students03 | 14029 | 140 | 62.491 | 1.089 |
| 29 | ucy_students03 | 14029 | 140 | 62.491 | 1.089 |
| 29 | ucy_students01 | 891 | 8 | 24.114 | 2.191 |
| 43 | ucy_students03 | 14029 | 140 | 62.491 | 1.089 |
| 43 | ucy_students01 | 891 | 8 | 24.114 | 2.191 |

The top-tail slice reads labels for diagnosis only, never inference.
The 0.01 scale slice is recording-local and cannot be pooled as a metric-unit threshold.
Three seeds share the same floor labels; repeated floor values are not independent evidence.
This diagnoses weighting sensitivity, not an error in the original annotations or a proven causal training failure.
