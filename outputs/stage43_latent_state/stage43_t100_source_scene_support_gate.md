# Stage43-CO T100 Source/Scene Support Gate

- source: `fresh_stage43_co_t100_source_scene_support_gate`
- result_source: `fresh_source_scene_supported_t100_gate_on_current_matrix`
- gate: `14 / 14`
- verdict: `stage43_co_t100_source_scene_support_gate_pass_floor_required`
- deploy t100: `False`

## Support Rule Protocol

- selection data: `validation_only`
- test threshold tuning: `False`
- min support rows: `200`
- rule: `switch_t100_only_if_source_file_or_scene_has_validation_positive_easy_safe_support`

## Current Support

- t100 test rows: `18070`
- source-supported t100 rows: `0`
- scene-supported t100 rows: `0`
- switched t100 rows: `0`
- blocked t100 rows: `18070`
- blocked t100 ratio: `100.00%`
- source-file overlap jaccard: `0.0000`
- scene overlap jaccard: `0.0000`

## Metrics

- raw family-rule t100 lift: `-3.86%`
- raw family-rule easy degradation: `2.26%`
- source/scene-supported all lift: `0.00%`
- source/scene-supported t100 lift: `0.00%`
- source/scene-supported hard/failure lift: `0.00%`
- source/scene-supported easy degradation: `0.00%`

## Worst Current Test Source Files

| source_file | rows | candidate t100 lift | easy harm | switch |
| --- | ---: | ---: | ---: | ---: |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/biwi/biwi_hotel.txt` | 1160 | -497.30% | 3007.96% | 0.00% |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt` | 15470 | -4.32% | 21.00% | 0.00% |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt` | 1440 | 0.80% | 18.93% | 0.00% |

## Interpretation

The stricter source-file/scene support gate blocks every current t100 test switch because no current t100 test source or scene has validation support. That is the correct safety behavior: the broader source-family rule was negative and easy-harmful on test.

This does not solve t100; it narrows the next requirement. Future t100 deployment needs validation support at source-file or scene granularity, or a new split/source acquisition that provides that support.

Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric or seconds-level claim; no Stage5C execution; no SMC.
