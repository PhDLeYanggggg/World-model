# Stage42-IK T50 Ensemble UCY Specialist Integration

- source: `fresh_stage42_ik_t50_ensemble_ucy_specialist_integration`
- generated_at_utc: `2026-05-28T01:59:57.914692+00:00`
- input_hash: `e9c42682df9cb49d952b7e3587f759a2f70204dbe5c95b76a7b92e7aa8b19a78`
- gate: `16 / 16`
- verdict: `stage42_ik_ucy_specialist_integration_pass`

## Purpose

Stage42-IJ showed that the Stage42-II t+50 ensemble is positive on TrajNet and ETH_UCY but remains fallback-only on UCY `crowds_zara03.txt`. Stage42-IK composes the verified Stage42-II non-UCY ensemble with the row-aligned Stage42-X UCY full-waypoint specialist.

This is not new training and not an independent new external-domain claim. It is a source-specialist composition test with strict row alignment and unchanged raw-frame / dataset-local boundaries.

## Claim Boundary

- dataset-local/raw-frame 2.5D only
- no metric or seconds-level claim
- no true 3D/foundation claim
- no Stage5C execution
- no SMC
- UCY repair is source-specialist evidence, not a new independent-domain proof

## Summary

| metric | value |
| --- | ---: |
| rows | 55528 |
| ADE all | 0.158819 |
| ADE t50 | 0.104522 |
| ADE t50 row CI low | 0.097328 |
| ADE t100 raw diagnostic | 0.180729 |
| ADE hard/failure | 0.163730 |
| ADE easy degradation | 0.000000 |
| FDE t50 | 0.263687 |
| FDE t50 CI low | 0.256358 |
| switch rate | 0.306440 |
| TrajNet t50 | 0.164168 |
| ETH_UCY t50 | 0.062896 |
| UCY t50 | 0.122892 |

## Per-Domain ADE

| domain | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 25901 | 0.101778 | 0.062896 | 0.102483 | 0.106084 | 0.005424 | 0.252152 |
| `TrajNet` | 20087 | 0.232755 | 0.164168 | 0.307144 | 0.238808 | 0.000000 | 0.282023 |
| `UCY` | 9540 | 0.196091 | 0.122892 | 0.213880 | 0.207360 | 0.000000 | 0.505241 |

## Source-File Rows

| source file | rows | t50 rows | all | t50 | hard/failure | easy degradation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara02.txt` | 20087 | 4927 | 0.232755 | 0.164168 | 0.238808 | 0.000000 |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/zara02/obsmat.txt` | 25901 | 6422 | 0.101778 | 0.062896 | 0.106084 | 0.005424 |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/zara03/crowds_zara03.txt` | 9540 | 2340 | 0.196091 | 0.122892 | 0.207360 | 0.000000 |

## Alignment

- alignment: `{'stage42x_ucy_rows': 9540, 'stage42ii_ucy_rows': 9540, 'ucy_mask_matches_domain': True, 'horizon_order_available': True, 'source_file_order_available': True, 'floor_ade_max_abs_delta': 1.5758981941615957e-07, 'floor_fde_max_abs_delta': 2.3840437091138256e-07}`

## Interpretation

- UCY is no longer fallback-only under this source-specialist composition.
- ETH_UCY and TrajNet keep the Stage42-II ensemble decisions; UCY uses the row-aligned Stage42-X full-waypoint specialist.
- This narrows the Stage42-IJ weak-source ledger, but it does not remove the need for future independent external-domain validation.
