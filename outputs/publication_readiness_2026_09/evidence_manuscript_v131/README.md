# Evidence Manuscript v131

This version records the signed risk retraining, recovered seed
opportunities, and the difference between known-step and conservative
effects at identical intervention counts. The supplement includes a
new vector figure generated from the verified scalar readouts.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v131
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v131.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v131.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `e353abf592636beea26f135801073a0d435d34139b0e737663cd5cc4588d5c9f` |
| `supplement/supplement.pdf` (38 pages) | `b208e8873dab7fe07aac336bcf11f5f265752ce15b30cf17a1250ab88bea8f5e` |
| `supplement/signed_coverage_ablation.pdf` | `7bdf4f79d3b0879210f0379ee9417f24968ac972ecf5de02bbaa216890e51937` |
| `main_source.zip` | `74e1ba9b0f71e24c56bc780da0fed31dfcf44d6c3b9a78811b08bf1e7a697bd2` |
| `supplement_source.zip` | `845675337d86d05c347dc79037eb14d2924e8f44a0f46a3e60e52af9df05550c` |

All 46 pages match clean source-archive builds at 85 DPI using the same
separately supplied style. The new figure and its supplement page
were also inspected visually. No PDF, archive, raw data, or model
weights are committed. The paper is not submitted or independently
confirmed.

[Signed risk training and replay](../moving_cv_signed_coverage_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
