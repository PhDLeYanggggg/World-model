# Frozen Appearance And Temporal Prediction: Completed Evidence

## Conclusion

All three fixed arms lose to stationary CV on the primary source cross-fit readout. This experiment does not establish deployable forecasting benefit.

All 36 fresh neural trajectory heads completed their registered 10,000 updates.
No model, seed, threshold, row subset or checkpoint was selected using held results.
The three arms are reported together. An oracle is a future-label diagnostic, not a policy.

## Scope And Provenance

- `fresh_run`: 25,300 frozen-image embeddings, 36 heads, 360,000 updates, all-arm analysis, replay and verification.
- `cached_verified`: supplied image/history corpus, registered source split, geometric normalizers and matched sampler controls.
- `not_run`: main benchmark, bookstore/outer scoring, final confirmation, Stage37 comparison, new deployment and strict sensor-as-of evaluation.
- 15,430 stationary-history queries, 29 recordings, 545 scoped agents, four historically explored source sites.
- Eight observed and twelve predicted annotation steps, stride 12 raw frames. This is not a raw-frame t+50 rerun.
- Coordinate errors remain annotation-pixel/local diagnostics. No seconds, metric, true 3D or foundation-model claim.
- The offline annotation contract allows supplied interpolated histories. It is not real-time perception certification or human motion gold.

## Fixed Comparison

All arms use the same geometry/coverage, projection/GRU/trajectory architecture,
three seeds and full-population sampling. The geometry arm zeroes visual features;
current appearance repeats the last image embedding; temporal appearance uses all
eight historical embeddings. ResNet18 remains frozen. Only the trajectory heads
are trained, each with parameter count [63960]. Source crops are 32 x 32 before
upsampling; interpolation does not restore lost visual detail.

| Arm | Equal-site ADE gain (%) | Conditional 95% CI | Window gain (%) | Easy harm (annotation px) | Positive held fits |
| --- | ---: | --- | ---: | ---: | ---: |
| Geometry + coverage | -0.070 | [-0.164, -0.014] | -0.100 | 0.001644 | 0/12 |
| Current appearance | -1.908 | [-2.942, -1.020] | -2.095 | 0.041487 | 0/12 |
| Eight-frame appearance | -6.102 | [-9.171, -3.807] | -6.506 | 0.120212 | 0/12 |

Primary gain is the ratio of equal-site mean errors, not a simple average of
site percentage gains. Seeds are averaged as errors, not prediction ensembles.
The 2,000 site-resamples are conditional on four explored sites with shared
training populations. These intervals are not independent confirmation.

| Fixed contrast | Gain difference (percentage points) | Conditional 95% CI |
| --- | ---: | --- |
| Eight-frame appearance minus Geometry + coverage | -6.032 | [-9.024, -3.794] |
| Eight-frame appearance minus Current appearance | -4.194 | [-6.229, -2.673] |
| Current appearance minus Geometry + coverage | -1.838 | [-2.796, -1.001] |

## Per-Site And Seed Results

| Site | Arm | Train gain (%) | Held gain (%) | Held hard gain (%) | Seed held gains (%) |
| --- | --- | ---: | ---: | ---: | --- |
| coupa | geometry | -0.013 | -0.015 | -0.001 | -0.014, -0.016, -0.016 |
| coupa | current | +0.015 | -0.747 | +0.001 | -0.806, -0.566, -0.868 |
| coupa | sequence | +0.188 | -3.153 | -0.027 | -3.502, -2.974, -2.982 |
| deathCircle | geometry | -0.013 | -0.012 | -0.000 | -0.011, -0.013, -0.011 |
| deathCircle | current | +0.121 | -1.499 | -0.011 | -1.425, -1.498, -1.573 |
| deathCircle | sequence | +0.550 | -4.422 | -0.030 | -4.763, -3.918, -4.585 |
| gates | geometry | -0.010 | -0.037 | +0.000 | -0.038, -0.031, -0.041 |
| gates | current | +0.034 | -2.365 | -0.052 | -2.281, -2.255, -2.561 |
| gates | sequence | +0.268 | -8.414 | -0.209 | -9.559, -8.026, -7.655 |
| hyang | geometry | -0.013 | -0.217 | -0.015 | -0.273, -0.164, -0.215 |
| hyang | current | +0.353 | -3.316 | +0.156 | -3.283, -3.281, -3.384 |
| hyang | sequence | +1.499 | -9.662 | +0.608 | -10.239, -9.064, -9.683 |

Hard subsets use training-complement error cutoffs. No future-defined slice is
an inference input. Easy targets have zero CV error, so percentage degradation
is undefined; absolute harm is reported and cannot be relabeled a 2% safety pass.

## Error Magnitudes And Oracle

| Arm | ADE (annotation px) | FDE (annotation px) | ADE p95 | ADE p99 | Nonzero-target gain (%) | Oracle gain (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| geometry | 1.12878 | 2.36593 | 4.08184 | 17.57033 | -0.019 | +0.008 |
| current | 1.15127 | 2.47245 | 4.08413 | 17.56624 | -0.052 | +0.400 |
| sequence | 1.20102 | 2.60617 | 4.10878 | 17.56869 | -0.588 | +1.056 |

Native-pixel aggregates are descriptive for this SDD subset, not cross-dataset
metric averages. Oracle gains assume access to future errors and are not deployable.

## Compute And Verification

Frozen feature extraction: 361.561 seconds summed across
215 immutable chunks. Head fitting: 739.023 summed seconds,
including the 100-update pilot. CPU four threads, inter-op one, DataLoader workers
zero in the native arm64 environment. Three exact encoder chunk replays and
36 exact train/held head replays passed. The completed-run resume added zero
updates and preserved 339 hashed artifacts.

Raw query/image alignment, training/held separation, target poisoning, matched
sampling streams, finite bounded outputs and out-of-fold scores were checked.
These checks establish implementation evidence, not predictive success.

## Research Boundary

No new deployment, Stage5C execution or SMC activation. Historical selector scores
remain exploratory under the lineage audit. A visual contrast alone does not prove
world dynamics or a scene-level intervention contribution. Remaining requirements
include a useful cross-site candidate, safe independently calibrated intervention,
full-population forecasting, external domains and unexposed confirmation.

See [the immutable registration](../source_pretrained_temporal_decision.md),
[machine-readable results](analysis.json), [verification](verification.json) and
[reproduction commands](reproducibility.md).
