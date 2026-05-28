# Stage42-IJ T50 Ensemble Source Robustness

- source: `fresh_stage42_ij_t50_ensemble_source_robustness`
- generated_at_utc: `2026-05-28T00:03:02.004847+00:00`
- input_hash: `989f16d5a2010f74e7b1b3b5dc337300c28aa2ff6d4b0f1dd27e8916475b90b3`
- gate: `15 / 15`
- verdict: `stage42_ij_t50_ensemble_source_robustness_pass`

## Purpose

Stage42-II repaired the t+50 seed-instability blocker with a validation-selected score/prediction ensemble. Stage42-IJ checks whether that result remains stable at source-file and scene levels.

## Summary

| metric | value |
| --- | ---: |
| rows | 55528 |
| source count | 3 |
| scene count | 3 |
| all improvement | 0.121192 |
| t50 improvement | 0.081363 |
| t50 source-group CI low | 0.000000 |
| t50 scene-group CI low | 0.000000 |
| hard/failure improvement | 0.124775 |
| easy degradation | 0.000000 |
| powered t50 sources positive / total | 2 / 3 |
| powered t50 scenes positive / total | 2 / 3 |

## Source-File Rows

| source file | rows | t50 rows | all | t50 | hard/failure | easy degradation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `crowds_zara02.txt` | 20087 | 4927 | 0.232755 | 0.164168 | 0.238808 | 0.000000 |
| `obsmat.txt` | 25901 | 6422 | 0.101778 | 0.062896 | 0.106084 | 0.005424 |
| `crowds_zara03.txt` | 9540 | 2340 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Interpretation

- This is a fresh source/scene robustness evaluation from cached-verified Stage42-II intermediates.
- No new model training is claimed.
- UCY remains fallback-only in Stage42-II; this audit does not rewrite fallback as positive transfer.
- Results remain dataset-local/raw-frame 2.5D only, with no metric or seconds-level claim.
