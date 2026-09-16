# Local External Source Availability Audit

Fresh raw-file parsing and identity/availability counting, not a training or forecasting result.
No coordinates were rescaled, no gaps interpolated, no official split assigned. No raw data exported.

| Source | Files | Parsed tracks | Unique agent-frame points | Recording units | obs8/pred12 views | Raw t50 views |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GC | 12,684 | 12,684 | 456,410 | 1 | 183,436 | 0 |
| HERMES | 51 | 9,844 | 2,757,226 | 51 | 2,570,190 | 2196118 |
| Wild-Track | 400 | 313 | 9,518 | 1 | 4,558 | 4926 |
| CITR | 344 | 344 | 95,648 | 38 | 89,112 | 76040 |
| VRU | 1,562 | 1,560 | 488,680 | None | 454,032 | not_run |

A recording unit is not proof of an independent physical scene, participant population, or untouched test source.
All window views overlap. These counts cannot be summed into an independent calibration sample size.
Availability uses the complete track for metadata only; track/future availability must not become inference features.
The proposed obs8/pred12 view is not yet an approved main protocol; raw horizons are not seconds.

## GC

- Result source: `fresh_run`; status: `metadata_audited_not_experiment_approved`.
- Scene support: One station recording; agent files are not independent scenes.
- Coordinate status: `native_image_pixels_axis_interpretation_unverified`; no verified metric or seconds claim.
- Terms: Local README: no license issued; exact release terms unresolved.
- Readiness: Parser available; release identity/terms/exposure review before any new use.
- Input manifest digest: `6542e286b5b3c8c89538a807b4874341d0d4bbe945afff85b5b2d734306f7a0f`.
- Track lengths (points): `{'min': 1.0, 'p25': 21.0, 'median': 34.0, 'p75': 45.0, 'p95': 69.0, 'max': 995.0}`.
- Continuous-run lengths: `{'min': 1.0, 'p25': 8.0, 'median': 21.0, 'p75': 37.0, 'p95': 52.0, 'max': 155.0}`.
- K=8/16/32/64 with 12 exact subsequent observation steps: `{'8': 183436, '16': 109551, '32': 23222, '64': 714}`.
- K=8 and exact raw t10/t25/t50/t100: `{'10': 0, '25': 0, '50': 0, '100': 267468}`.
- Discontinuous edges: 6824; conflicting/invalid tracks or files: 0.
- Byte-identical excess files: 0; geometry-identical excess tracks: 0 (counts not automatically promoted or treated as independent).

## HERMES

- Result source: `fresh_run`; status: `metadata_audited_not_experiment_approved`.
- Scene support: Controlled corridor trials; shared setting/participants, not independent natural sites.
- Coordinate status: `native_xy_source_claims_centimeters_not_verified`; no verified metric or seconds claim.
- Terms: Exact local experiment terms unresolved; archive access not verified.
- Readiness: Possible controlled stress/curriculum support, not natural-domain success.
- Input manifest digest: `86d99af62b1395776d5523a92a7833a8399880b475e21c433a144a56eae5b5e8`.
- Track lengths (points): `{'min': 102.0, 'p25': 172.0, 'median': 216.0, 'p75': 328.25, 'p95': 618.0, 'max': 1220.0}`.
- Continuous-run lengths: `{'min': 102.0, 'p25': 172.0, 'median': 216.0, 'p75': 328.25, 'p95': 618.0, 'max': 1220.0}`.
- K=8/16/32/64 with 12 exact subsequent observation steps: `{'8': 2570190, '16': 2491438, '32': 2333934, '64': 2018926}`.
- K=8 and exact raw t10/t25/t50/t100: `{'10': 2589878, '25': 2442218, '50': 2196118, '100': 1703924}`.
- Discontinuous edges: 0; conflicting/invalid tracks or files: 0.
- Byte-identical excess files: 0; geometry-identical excess tracks: 0 (counts not automatically promoted or treated as independent).

## Wild-Track

- Result source: `fresh_run`; status: `metadata_audited_not_experiment_approved`.
- Scene support: Seven synchronized cameras share one scene; not seven domains.
- Coordinate status: `native_integer_ground_grid_not_meters`; no verified metric or seconds claim.
- Terms: Original page inspected, dataset-specific terms not established.
- Readiness: Previous diagnostic intake found; conversion/terms unresolved.
- Input manifest digest: `47671489cebcf1fdc6d3b54bc97dbb5db00e01773fbfd14f591dd30f35aafc94`.
- Track lengths (points): `{'min': 1.0, 'p25': 12.0, 'median': 21.0, 'p75': 29.0, 'p95': 60.39999999999998, 'max': 330.0}`.
- Continuous-run lengths: `{'min': 1.0, 'p25': 10.0, 'median': 19.0, 'p75': 27.0, 'p95': 56.69999999999999, 'max': 330.0}`.
- K=8/16/32/64 with 12 exact subsequent observation steps: `{'8': 4558, '16': 3509, '32': 2566, '64': 1963}`.
- K=8 and exact raw t10/t25/t50/t100: `{'10': 6753, '25': 5997, '50': 4926, '100': 3509}`.
- Discontinuous edges: 34; conflicting/invalid tracks or files: 0.
- Byte-identical excess files: 0; geometry-identical excess tracks: 0 (counts not automatically promoted or treated as independent).

## CITR

- Result source: `fresh_run`; status: `metadata_audited_not_experiment_approved`.
- Scene support: Controlled clips at one parking lot; six scenarios are not six independent sites.
- Coordinate status: `dataset_local_xy_source_claims_meters_not_verified`; no verified metric or seconds claim.
- Terms: Author repository inspected; local release permissions unresolved.
- Readiness: Native synchronized clips usable by parser; terms/exposure and grouping pending.
- Input manifest digest: `3deda5d8c0b421587a924b2d702d512e5ea1afbca5940fee1193c12bff5b4088`.
- Track lengths (points): `{'min': 154.0, 'p25': 221.0, 'median': 290.0, 'p75': 320.0, 'p95': 381.0, 'max': 421.0}`.
- Continuous-run lengths: `{'min': 154.0, 'p25': 221.0, 'median': 290.0, 'p75': 320.0, 'p95': 381.0, 'max': 421.0}`.
- K=8/16/32/64 with 12 exact subsequent observation steps: `{'8': 89112, '16': 86360, '32': 80856, '64': 69848}`.
- K=8 and exact raw t10/t25/t50/t100: `{'10': 89800, '25': 84640, '50': 76040, '100': 58840}`.
- Discontinuous edges: 0; conflicting/invalid tracks or files: 0.
- Byte-identical excess files: 0; geometry-identical excess tracks: 0 (counts not automatically promoted or treated as independent).

## VRU

- Result source: `fresh_run`; status: `partial_or_missing_source_review_required`.
- Scene support: Single-object files with local clocks; global recording/neighbor alignment unknown.
- Coordinate status: `dataset_local_xy_source_claims_meters_not_verified`; no verified metric or seconds claim.
- Terms: Source page inspected; newer extended release terms cannot be inherited.
- Readiness: Single-agent history possible; synchronized multi-agent use is blocked.
- Input manifest digest: `e0e7eb57ce35ecadc5a9f50447f7a2cd5e68976b98f1ab031827b711519e344e`.
- Track lengths (points): `{'min': 49.0, 'p25': 228.75, 'median': 302.5, 'p75': 388.0, 'p95': 502.0, 'max': 1257.0}`.
- Continuous-run lengths: `{'min': 1.0, 'p25': 121.0, 'median': 247.0, 'p75': 349.75, 'p95': 501.0, 'max': 1257.0}`.
- K=8/16/32/64 with 12 exact subsequent observation steps: `{'8': 454032, '16': 440009, '32': 412504, '64': 359252}`.
- K=8 and exact raw t10/t25/t50/t100: `None`.
- Discontinuous edges: 430; conflicting/invalid tracks or files: 2.
- Byte-identical excess files: 0; geometry-identical excess tracks: 0 (counts not automatically promoted or treated as independent).

## Limits

DUT has reference material only in the inspected local directory; no trajectory audit or new download was performed.
Prior predictive exposure remains unknown unless separately established. Wild-Track has an existing diagnostic intake record.
Source conditions and independent-scene grouping must be settled before adding any source to a formal experiment.
See `source_review.md` for original-source links, release mismatches, loader issues and next actions.
No Stage5C execution, SMC, deployment promotion, or new real-world predictive gain.
