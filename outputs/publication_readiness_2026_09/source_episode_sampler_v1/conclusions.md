# Equal-Episode Exposure: Completed Comparison

## Conclusion

Equal-episode training still does not beat stationary CV. No new deployment.

All 24 fresh heads completed 10,000 updates each. The two original uniform
controls (24 heads) are cached_verified, not retrained. No held-driven stopping,
seed, checkpoint, threshold or population selection. Registration commits
662dcbba/dd5c7128 precede fitting; the latter fixes trailing whitespace and hashes.

## What Changed

Only training row probabilities changed: equal mass per past-defined annotation
episode within each training complement, then equal mass among its rows.
All rows keep positive probability. The per-example loss is unchanged, but the
expected training objective is reweighted; this is not an unbiased uniform-risk
sampler. Geometry/coverage, frozen visual features,
63,960 parameters, seeded initialization, all-target ADE, original normalizer,
training cost scale, hard cutoff and original evaluation are unchanged.
The geometry and centered arms share weighted draws; draws intentionally differ
from uniform controls. No static-gradient removal or future-label group key.

Same 15,430 source queries, four explored sites, three seeds, eight observed and
twelve predicted annotation steps at stride 12 raw frames. Not the main benchmark
or t+50 supplement. Past-indexed supplied annotations may use later interpolation
controls: offline, not strict sensor-as-of. No metric or seconds equivalence.

## Primary Results

| Arm | Source | Equal-site gain (%) | Conditional site interval | Absolute static harm (px) | Positive held fits |
| --- | --- | ---: | --- | ---: | ---: |
| geometry_uniform | cached_verified_recomputed | -0.0704 | [-0.1644, -0.0135] | 0.001644 | 0/12 |
| centered_uniform | cached_verified_recomputed | -0.7622 | [-1.7428, -0.1231] | 0.020320 | 0/12 |
| geometry_event | fresh_run | -37.3268 | [-52.3613, -26.0828] | 0.576017 | 0/12 |
| centered_event | fresh_run | -54.9917 | [-79.7777, -39.1095] | 0.765669 | 0/12 |

| Fixed contrast | Difference (pp) | Conditional interval |
| --- | ---: | --- |
| geometry_event minus geometry_uniform | -37.2564 | [-52.2149, -26.0693] |
| centered_event minus centered_uniform | -54.2295 | [-79.2560, -38.9263] |
| centered_event minus geometry_event | -17.6649 | [-31.0197, -6.9387] |

Primary remains the ratio of equal-site mean normalized errors, not mean site
percentages or episode-reweighted evaluation. Seeds average errors, not predictions.
The 2,000 shared site-bootstrap draws remain conditional on four explored sites
and overlapping fitting complements. They are not independent confirmation.

## Every Site And Seed

| Arm | Site | Training gain (%) | Held gain (%) | Hard gain (%) | Three held seed gains (%) |
| --- | --- | ---: | ---: | ---: | --- |
| geometry_uniform | coupa | -0.0126 | -0.0154 | -0.0006 | -0.0139, -0.0159, -0.0163 |
| geometry_uniform | deathCircle | -0.0134 | -0.0118 | -0.0003 | -0.0114, -0.0129, -0.0111 |
| geometry_uniform | gates | -0.0095 | -0.0369 | +0.0000 | -0.0379, -0.0314, -0.0414 |
| geometry_uniform | hyang | -0.0132 | -0.2174 | -0.0148 | -0.2735, -0.1638, -0.2149 |
| centered_uniform | coupa | -0.0138 | -0.0519 | -0.0010 | -0.0467, -0.0370, -0.0719 |
| centered_uniform | deathCircle | -0.0084 | -0.2243 | +0.0023 | -0.1725, -0.2078, -0.2926 |
| centered_uniform | gates | -0.0140 | -0.4750 | +0.0180 | -0.6276, -0.2739, -0.5236 |
| centered_uniform | hyang | +0.2371 | -2.3182 | +0.0101 | -2.1318, -2.3825, -2.4403 |
| geometry_event | coupa | -11.4977 | -24.6548 | +0.1685 | -24.5408, -23.8801, -25.5434 |
| geometry_event | deathCircle | -14.0630 | -27.4239 | -0.1284 | -25.9051, -28.4622, -27.9045 |
| geometry_event | gates | -10.8813 | -60.2917 | +0.3293 | -56.8874, -62.1025, -61.8854 |
| geometry_event | hyang | -13.4398 | -47.2172 | -1.8436 | -48.9020, -48.5918, -44.1579 |
| centered_event | coupa | -18.0981 | -54.1342 | -2.0312 | -54.1070, -52.1355, -56.1600 |
| centered_event | deathCircle | -19.2352 | -34.4063 | -0.6075 | -33.3473, -34.3956, -35.4759 |
| centered_event | gates | -16.5122 | -93.8466 | -1.1990 | -91.2186, -96.3894, -93.9319 |
| centered_event | hyang | -14.9960 | -54.1063 | -2.3531 | -55.2209, -53.7219, -53.3760 |

## Error Magnitude

| Arm | Native ADE | Native FDE | Nonzero gain (%) | p95 ADE | p99 ADE | Binary oracle (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| geometry_uniform | 1.128781 | 2.365927 | -0.0191 | 4.081840 | 17.570326 | 0.007628 |
| centered_uniform | 1.139625 | 2.407879 | -0.0614 | 4.083270 | 17.570309 | 0.163048 |
| geometry_event | 1.546256 | 3.036765 | -8.7639 | 4.585501 | 17.641198 | 2.605237 |
| centered_event | 1.719345 | 3.262957 | -14.7767 | 4.748245 | 17.654755 | 3.281613 |

Native errors are annotation-pixel descriptions. Static baseline error is zero;
percentage easy degradation is undefined, not a 2% safety pass. Future-informed
binary oracles are label-only diagnostics, not a learned switching policy.

## Reproducibility

Fresh fitting: 240,000 updates, 531.085s summed fitting including the pilot.
Native arm64 CPU, four compute threads, one inter-op thread, zero loader workers.
All 24 forecasts replay exactly; weighted draws are independently regenerated.
All training group masses verify, twelve paired-arm streams match, six OOF
archives recompute. Completed resume adds zero updates and preserves
84 artifacts. No raw data, feature cache or checkpoints in Git.

Fresh: weighted fitting, analysis and artifact verification. Cached_verified:
episode identities, image features and uniform controls. Not_run: independent
confirmation, main/outer and external forecasts, new intervention policy.
No new deployment, Stage5C execution or SMC. Not submission-ready.

See [fixed design](../source_episode_sampler_decision.md), [analysis](analysis.json),
[verification](verification.json), [failure analysis](failure_analysis.md)
and [evidence gates](gates.md).
