# Evidence Gates

Engineering completion and scientific success are separate.

| Gate | Status | Evidence |
| --- | --- | --- |
| Current-cohort event audit | Complete | 15,430 rows, 1,457 past-defined groups, exact raw rerun and 1,077 future mutation/truncation checks. |
| Training preregistration | Pass | 662dcbba/dd5c7128 before fitting; fixed 24 heads and 240,000 updates. |
| Actual neural training | Complete | All 24 heads finish 10,000 updates; image encoder remains frozen. |
| Training-only sampling | Pass for checked artifacts | Past group keys; group frequencies only inside each training complement; every row has positive probability. |
| Matched training controls | Pass | Same data, per-row loss, architecture, initialization and normalizers; sampler changes the expected training objective explicitly. |
| Reproducibility | Pass locally | 24 exact replays and regenerated draws, 12 paired streams, six recomputed OOF archives, 84 artifacts unchanged on zero-update resume. |
| Better original held-site forecasts | Fail | Geometry -37.3268%; centered -54.9917%; all new held fits negative. |
| Improvement over uniform controls | Fail | -37.2564pp and -54.2295pp. |
| Added visual contribution | Fail | Centered minus geometry -17.6649pp; conditional interval below zero. |
| Easy preservation at 2% | Not established | Zero-error floor makes percentage undefined; disclosed absolute harms increase substantially. |
| Hard or moving-target benefit | Fail as a robust claim | Moving targets worsen; all centered hard slices negative. |
| Independent calibration/confirmation | Not run | Four explored sites/shared folds only. |
| Main/t+50/external/Stage37 comparison | Not run | This source subset is not a replacement benchmark. |
| Deployment/publication candidate | No | Predictive benefit and independent evidence remain missing. |
| Stage5C | False / forbidden | Not executed. |
| SMC | False / forbidden | Not enabled. |

Fresh: event audit, 24 weighted fits, analysis, objective-shift diagnosis and
verification. Cached_verified: old uniform controls, image features and source
provenance. No importance-corrected follow-up training has run. All required
processes are terminal. Twenty-eight scoped tests pass; no repository-wide test
claim. CPU arm64, four compute threads, one inter-op thread, zero loader workers.
Summed fitting 531.085s including pilot; main-log span 537.627s. No CREATE job.
