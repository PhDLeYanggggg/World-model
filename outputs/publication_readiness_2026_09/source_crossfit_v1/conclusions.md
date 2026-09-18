# Candidate Cross-Fitting: Completed, No Useful Out-of-Site Advantage

## Material Passport

`fresh_run`: twelve cold-start Torch models, four internal held sites and three
seeds, 120,000 optimizer updates. Each model has 44,864 parameters. The full-run
log spans 5,784.64 seconds (96.41 minutes); summed fitting time, including the
100-update pilot, is 5,725.36 seconds. No budget reduction or early stopping.

`cached_verified`: admitted source data, shared code and the descriptive
full-four-site in-sample reference. Old fitted parents were not reused.
`not_run`: a new risk head, outer-bookstore inference, main evaluation,
independent calibration or confirmation, deployment and new cross-dataset claims.

This is the complete stationary-history training-side subset: 15,430 rows,
29 recordings, 545 recording-scoped agents and four physical sites. It is not
the full mixed-motion benchmark. All sites were explored previously. Predictions
are OOF by physical site, but are not independent confirmation observations.

## Results

Positive gain means lower ADE than stationary CV. Seed summaries average errors,
not forecast trajectories. The primary ratio uses equal-site mean normalized
ADE. Conditional intervals use 2,000 resamples; fold-training sets overlap.

| Inner held site | Rows | Training-complement mean gain | OOF gain | Conditional recording 95% interval |
| --- | ---: | ---: | ---: | --- |
| coupa | 4,250 | +0.7951% | -2.1932% | [-3.3252%, -1.7229%] |
| deathCircle | 3,054 | +1.0821% | -2.7643% | [-3.5426%, -1.3310%] |
| gates | 1,662 | +0.5440% | -7.7589% | [-13.3678%, -3.1614%] |
| hyang | 6,464 | +2.5438% | -8.8101% | [-17.5926%, -4.9037%] |

**Primary equal-site gain: -5.01598%, conditional four-site interval
[-8.39655%, -2.48773%].** Every one of the twelve held-site/seed fits is negative.
Window-weighted sensitivity is -5.45692%; seed gains are -5.61106%, -6.07979%
and -4.67991%. Native annotation-pixel ADE is 1.18919 versus CV 1.12765.
These conditional intervals describe the registered training diagnostic; they
are not independent-scene safety guarantees or a new final-test result.

Hard-slice gains by site are -0.01223%, -0.01155%, +0.11048% and -0.20061%,
using each producer's training-only cutoff. No consistently improved hard slice.
There are 8,566 zero-target queries. Their absolute mean harm is 0.09190 pixels;
percentage easy degradation is undefined because CV error is zero, not a passed
2% preservation gate.

## Why This Changes the Next Step

The registered binary oracle chooses CV or that seed's candidate using future
truth. Its window-weighted gain is only **0.52707%**. A separately marked
post-hoc calculation gives **0.46765%** under the primary equal-site estimator.
This is a labeled-population upper bound for this fixed two-action class, not
a learnable policy, a neural information bound or a bound on other predictors.
Training a larger selector on the same outputs cannot manufacture a large
improvement absent from that action class.

The post-hoc attribution splits the primary 5.01598 percentage points of excess
error into **4.32112pp on zero-target rows** and **0.69486pp on nonzero-target
rows**. On nonzero targets alone the candidate still loses 0.93253% in the
window-weighted view. Thus false movement dominates the measured harm, but
removing it with a perfect label-informed guard would not fix the remaining
candidate's average error. About 41.1% of nonzero queries benefit per seed;
those benefits are too small to establish a useful overall action family.

The cached all-four-site in-sample reference gains +0.18724%. Its benefit sign
disagrees with OOF on 17.6993% of row/seed pairs; 14.0268% change from reference
benefit to non-benefit. This supports treating in-sample cost labels as an
unreliable substitute for the measured OOF costs. It does **not** isolate
training optimism as a cause: training population, site exposure and fitted
normalizers all change together.

## Failure Taxonomy

| Explanation | Evidence and limit |
| --- | --- |
| Runtime or incomplete training | Not supported: full budget, finite bounded predictions, exact replays and zero-update resume verified |
| Training fit versus site transfer | All twelve train gains positive and held gains negative; a clear measured gap, not a causal attribution to one mechanism |
| Unsafe invented movement | Supported on zero-target rows: 4.32112pp of primary excess error |
| Only the selector threshold is wrong | Insufficient explanation: ungated fixed-candidate oracle headroom is below 0.53% in both reported weightings |
| Candidate motion is sufficiently useful once easy rows are removed | Not established: nonzero-target candidate gain is -0.93253% |
| Input information or annotation resolution is insufficient | Remains a hypothesis; this experiment does not prove irreducibility, strict real-time availability or a causal information limit |
| Multimodal/JEPA/Transformer contribution | Not tested here: the fixed arm masks RGB and reuses a small geometry/coverage predictor |

## Verification and Decision

Twelve models replay both training and held predictions exactly. The independent
verifier checks twelve cold-start producer lineages, three exact-once OOF
archives, 96 loaded-future-target poisoning checks and 55 immutable artifacts
under a zero-update completed resume. These checks establish the stated
implementation contract, not raw-label acquisition causality or prediction gain.
Twenty-eight focused tests pass. The non-hermetic legacy full suite is not rerun.
The generated comparison figure was visually inspected.

Do not promote a model, retune bookstore or open sealed main roles. The next
candidate work must distinguish start/movement detection from direction and
path prediction using only admitted training material, with fixed causal-input
controls and producer lineage. Before more risk-head fitting, require useful
OOF candidate utility; any risk-head validation needs nested producer exclusions.
That follow-up has not been trained here. No Stage5C execution or SMC.

The project remains a 2.5D research system, not a demonstrated foundation or
true-3D model. Results are offline annotation pixels/past-normalized raw frames,
not meters, seconds or human gold. Submission readiness remains unmet.
