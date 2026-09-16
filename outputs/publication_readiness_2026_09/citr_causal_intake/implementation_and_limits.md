# CITR Causal Diagnostic Intake

Date: 2026-09-16. This is a real-data conversion and input-integrity result, not a forecasting experiment or an approved benchmark split.

## What Was Run

The local OpenTraj CITR raw-object CSV files were converted into the existing recording-centric causal schema. Their paths and Git blob hashes match the previously verified author snapshot, commit `b6a980628f48e7c80c748c3e333979ca3a6b0301`. The source-manifest identity was recomputed before and after conversion. No raw data were downloaded, modified or committed.

| Quantity | Fresh conversion |
| --- | ---: |
| Clip recordings | 38 |
| Physical-site groups | 1 |
| Raw object files / clip-scoped tracks | 344 |
| Pedestrian / vehicle tracks | 318 / 26 |
| Raw positions preserved and traced individually | 95,648 |
| Exact raw-frame t10 windows, K=8 | 89,800 |
| Exact raw-frame t25 windows, K=8 | 84,640 |
| Exact raw-frame t50 windows, K=8 | 76,040 |
| Exact raw-frame t100 windows, K=8 | 58,840 |
| Eight-observation / twelve-step windows | 89,112 |
| Conversion time, excluding subsequent input mutation check | 2.415 s |
| Cache size at build report | 20,857,198 bytes |

Window counts include both object types and overlap heavily. Different horizon views are not independent observations. Track IDs are clip-scoped; the count is not a count of distinct people or vehicles across trials. The [author repository](https://github.com/dongfang-steven-yang/vci-dataset-citr) describes controlled experiments at one parking lot. Accordingly, all clips share `citr_osu_parking_lot`; scenario and clip names do not create additional independent locations. The release README's advertised pedestrian count is not substituted for the locally verified raw count.

## Reader and Provenance

- Only raw `p<ID>.csv` and `v<ID>.csv` schemas are accepted. Filtered trajectories and published velocity fields are not used.
- Pedestrian `(x,y)` and vehicle `(x_c,y_c)` positions are retained exactly. The per-clip namespace is `2*source_id + type_code`, so `p1` and `v1` cannot collide.
- `source_rows.npy` maps every converted position to its source-file index and CSV line. All 95,648 rows were additionally reconstructed from CSV fields in a separate verification pass, without calling the converter parser. All stored endpoint frame differences match their indexed horizons.
- Missing frames are neither filled nor crossed to manufacture labels. Invalid raw rows, duplicate frame IDs, nonfinite positions, schema drift and source/cache hash changes are refused.
- `get_scene_inputs` selects current agents using only past support, including those without complete future labels. Future labels remain behind a separate API. No goals, central velocities, inherited teacher outputs, train/test statistics or track-remaining features were added.
- Agent types are exposed as contextual metadata. The existing neural tensor packer has not been extended to train typed embeddings in this slice. Retaining vehicle types is not evidence of vehicle-model performance.
- Input mutation checks used one index-eligible raw50 query per clip: 38 queries / 344 agents. Replacing every post-current position left all inputs, coordinate transforms and agent membership unchanged. No future-label API was called and no prediction accuracy was computed. This is a targeted input-boundary check, not proof of an entire future training pipeline's leakage freedom.

## Recovery and Result Sources

The build uses memory-mappable NPY arrays plus JSON metadata, a source-row map, a run identity and a completion receipt. A clip's arrays, metadata and receipt are committed by one directory rename. PID/progress heartbeats are written after each clip. Resume verifies the source manifest, converter and reader code, cache arrays, metadata and raw-row alignment before reusing a completed clip. An interrupted partial clip can be rebuilt only if its ownership marker matches this exact conversion. An unrelated partial directory is not deleted.

- [build_report.json](build_report.json): `fresh_run`, 38 new conversions and fresh causal-input checks.
- [resume_report.json](resume_report.json): `cached_verified`, all 38 converted clips reused; the input mutation check was rerun and is separately marked fresh.
- [verification.json](verification.json): focused tests, full row verification, unchanged protocol identity and scope limits.

The first test collection exposed an indentation error in the new module. It was corrected before conversion; the subsequent tests and real conversion passed. Tests also inject interrupted partial writes and post-commit interruption, verify same-run recovery, and reject tampered source/metadata/arrays/receipts.

## What This Does Not Establish

All caches remain `diagnostic_only`. The existing formal protocol is unchanged and unapproved. Source-use conditions, prior predictive exposure and suitability for independent confirmation remain unresolved. A local copy and an author-content hash do not establish permission, lack of prior exposure, or statistical independence. No clip has been assigned to training, model selection, calibration or confirmation.

The author describes coordinates as meters, but this intake does not independently verify calibration, scale correspondence or effective time. Coordinates remain `dataset_local_unverified`; horizons remain raw frames or observation steps, not seconds. Body extents, image alignment and physical safety are not validated.

No model was fitted on CITR; no baseline accuracy, learned gain, confidence interval or deployment result was computed. CITR supplies synchronized controlled examples for potential mechanism diagnostics, not the missing broad independent-scene confirmation evidence. The previously rebuilt formal-candidate catalog remains nine recordings / six physical-site groups; this separate one-site cache has not been merged into it.

DUT's author metadata was reviewed as another potential source. Raw download, conversion and model evaluation were not run; source-use review and an approved role remain pending. CREATE access is still unresolved, and no remote job was submitted. Stage5C and SMC remain off. This slice does not establish CVPR submission readiness.

## Next Dependency

Resolve the already-pending scientific decisions about observation/prediction horizon, data roles, aggregation and empirical versus formal risk claims. Then specify which controlled diagnostics CITR may support and which genuinely independent scenes can support confirmation. Do not turn 38 repeated clips into 38 independent calibration sites to bypass the support deficit.
