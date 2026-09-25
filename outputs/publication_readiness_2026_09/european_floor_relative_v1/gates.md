# Evidence Gates

Engineering completion and scientific efficacy are intentionally separate.
This table is not a count of stages that makes the project submission-ready.

| Check | Status | Evidence |
|---|---|---|
| Registration before fitting/readout | Pass | 1049876b pushed before pilot |
| Fresh full registered controller training | Pass | 234 heads, 468,000 updates; no new forecaster |
| Cross-fitted floor target exclusion | Pass within registered source scope | 2+2 fitting localities; all 8 outer localities excluded |
| Matched inputs, sampling and compute | Pass | 380 features; 36 matched sampler/preprocess groups |
| Both decision banks before readout | Pass | Frozen 13:39:35/13:40:56 UTC; first evaluation 13:41:16 UTC |
| Numerical/checkpoint reproducibility | Pass | 234 prefix replays; 2,160 alternate metric reductions; 180 old metrics exact |
| Scoped implementation tests | Pass | 301 tests in 47 files; full legacy suite not_run |
| Both-floor target advantage vs matched CV target | Fail | 33/36 points negative; 20 negative/0 positive all-ADE intervals |
| Incremental all-ADE over protected floor | Positive development evidence | 36/36 both-floor conditional intervals positive |
| Hard preservation in every locality | Fail | Worst losses 2.8055%/1.8413% despite positive average hard gains |
| Positive-easy worst-locality degradation <=2% | Observed pass | All 144 new-policy views; worst 0.3001% |
| Zero-reference preservation | Fail / weak support | Each arm harms 24/36 views; 12 others lack examples |
| Independent risk calibration/confirmation | not_run |Reserved roles closed; no certified safety |
| New dynamics training/lift | not_run |Frozen forecaster; new intervention heads only |
| Deployable neural promotion | No |No arm promoted; existing deployment unchanged |
| Submission readiness | No |Independent evidence, calibration and full main-method comparisons remain |
| Stage5C execution | False |Forbidden; not run |
| SMC | False |Forbidden; not enabled |

The population is 12 opened development localities, not 36 independent experiments
or 318,969 independent samples. Three seeds and 3,000 locality-bootstrap resamples
describe conditional uncertainty, not simultaneous protection. Raw annotation
obs8/pred12 stride12 in image pixels, automatic released tracks; no metric,
seconds, human-gold, physical-safety, true3D or foundation claim.
