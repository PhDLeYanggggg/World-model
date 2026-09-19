# Importance-Corrected Exposure: Completed Comparison

## Conclusion

Neither corrected arm beats stationary CV. No new deployment.

24 fresh heads complete 240,000 updates. Four original uniform/uncorrected arms
(48 heads) are cached_verified, not retrained. Registration b2276809 was pushed
before the included 100-update pilot. No held-driven stopping, checkpoint, seed,
threshold or cohort selection. All planned contrasts are retained.

## Intervention And Objective

Same episode sampler, draw streams, geometry/centered inputs, 63,960 parameters,
all rows, initializations, learning-rate schedule, normalizers and original metric.
Each sampled row loss is multiplied by 1/(N_train*p_train(row)); no clipping or
self-normalization of weights. Four training-fold fixed-offset probes verify
expected loss/unclipped-gradient equality. This is not an unbiased Adam-update
claim: clipping, adaptive optimization and finite-batch variance remain.

Same 15,430 source queries, four explored sites, three seeds, offline eight past
and twelve future annotation steps at stride 12 raw frames. Not main/t+50/external
evaluation. Past labels can have later interpolation controls; not sensor-as-of.
No verified metric/seconds scale, independent physical events or human-gold claim.

## Primary Results

| Arm | Source | Equal-site gain (%) | Conditional 95% interval | Static harm (annotation px) | Positive held fits |
| --- | --- | ---: | --- | ---: | ---: |
| geometry_uniform | cached_verified_recomputed | -0.0704 | [-0.1644, -0.0135] | 0.001644 | 0/12 |
| centered_uniform | cached_verified_recomputed | -0.7622 | [-1.7428, -0.1231] | 0.020320 | 0/12 |
| geometry_event | cached_verified_recomputed | -37.3268 | [-52.3613, -26.0828] | 0.576017 | 0/12 |
| centered_event | cached_verified_recomputed | -54.9917 | [-79.7777, -39.1095] | 0.765669 | 0/12 |
| geometry_corrected | fresh_run | -0.0321 | [-0.0577, -0.0128] | 0.000639 | 0/12 |
| centered_corrected | fresh_run | -0.2746 | [-0.6804, -0.0318] | 0.007827 | 0/12 |

| Fixed contrast | Gain difference (pp) | Conditional interval |
| --- | ---: | --- |
| geometry_corrected minus geometry_uniform | +0.0383 | [-0.0008, +0.1087] |
| centered_corrected minus centered_uniform | +0.4876 | [+0.0846, +1.0688] |
| geometry_corrected minus geometry_event | +37.2947 | [+26.0700, +52.3035] |
| centered_corrected minus centered_event | +54.7171 | [+39.0737, +79.6961] |
| centered_corrected minus geometry_corrected | -0.2425 | [-0.6248, -0.0189] |

Primary is the ratio of equal-site mean normalized errors. Seed errors are
averaged, not prediction-ensembled. The 2,000 site-bootstrap draws are conditional
on four explored sites/shared fitting folds, not independent confirmation.

## All Sites And Seeds

| Arm | Site | Training gain (%) | Held gain (%) | Hard gain (%) | Held seeds (%) |
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
| geometry_corrected | coupa | -0.0131 | -0.0149 | -0.0005 | -0.0160, -0.0124, -0.0164 |
| geometry_corrected | deathCircle | -0.0133 | -0.0109 | -0.0003 | -0.0110, -0.0084, -0.0132 |
| geometry_corrected | gates | -0.0133 | -0.0387 | -0.0006 | -0.0348, -0.0352, -0.0461 |
| geometry_corrected | hyang | -0.0183 | -0.0701 | -0.0036 | -0.1267, -0.0375, -0.0461 |
| centered_corrected | coupa | -0.0149 | -0.0233 | -0.0006 | -0.0265, -0.0212, -0.0221 |
| centered_corrected | deathCircle | -0.0206 | -0.0398 | +0.0001 | -0.0579, -0.0241, -0.0373 |
| centered_corrected | gates | -0.0173 | -0.1136 | +0.0025 | -0.1066, -0.0811, -0.1532 |
| centered_corrected | hyang | +0.0084 | -0.9143 | +0.0017 | -0.9662, -0.7312, -1.0453 |

## Error Magnitude

| Arm | Native ADE | Native FDE | Nonzero gain (%) | p95 ADE | p99 ADE | Binary oracle (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| geometry_uniform | 1.128781 | 2.365927 | -0.0191 | 4.081840 | 17.570326 | 0.007628 |
| centered_uniform | 1.139625 | 2.407879 | -0.0614 | 4.083270 | 17.570309 | 0.163048 |
| geometry_event | 1.546256 | 3.036765 | -8.7639 | 4.585501 | 17.641198 | 2.605237 |
| centered_event | 1.719345 | 3.262957 | -14.7767 | 4.748245 | 17.654755 | 3.281613 |
| geometry_corrected | 1.128088 | 2.362917 | -0.0071 | 4.081907 | 17.570362 | 0.002304 |
| centered_corrected | 1.132203 | 2.382693 | -0.0183 | 4.078854 | 17.570363 | 0.064806 |

Static CV error is zero; percentage easy degradation is undefined, not a 2% pass.
Native errors are annotation-pixel descriptions. Binary oracles use targets
only for diagnosis and do not demonstrate a learned switching policy.

## Verification And Cost

All 24 predictions replay exactly and draw streams regenerate. All 24 match their
uncorrected controls' draws; twelve paired-arm streams agree. Factors match the
training-only propensities. Six OOF archives recompute. Zero-update resume
preserves 84 immutable artifacts. 37 scoped tests pass;
no full legacy-suite rerun. Native arm64 CPU, four threads, one inter-op thread,
zero workers. Summed fitting
673.584s including pilot. Frozen image encoder is not retrained.

Fresh: expectation checks, corrected fits, analysis and verification.
Cached_verified: old controls, event groups, image features and source provenance.
Not_run: independent calibration/confirmation, main/outer/external readout,
new intervention policy. No new deployment, Stage5C execution or SMC.

See [design](../source_importance_sampling_decision.md), [analysis](analysis.json),
[verification](verification.json), [failure analysis](failure_analysis.md),
[gates](gates.md) and [reproduction](reproducibility.md).
