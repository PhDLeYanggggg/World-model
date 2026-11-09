# Evidence manuscript v163

This version adds the [full-path selection experiment](../camera_full_path_selection_v1/README.md).
The exposed camera has feasible hindsight plans, but the
source-excluded donor-trained policy abstains. The signed-risk
repair intervenes and still fails. The same-count, same-peak
joint and independent routes give identical paths in the
expanded menu. None of this is a deployment or confirmation pass.
The print-friendly figures from v161 remain unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v163
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v163.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v163.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `0b1f22a7ada66f9999ece7cc191f459b636275f2b05c8f7a4214f8a0c7538ec3` |
| `supplement/supplement.pdf` (50 pages) | `b46d471bf63ba96dfb82360661e28dabfd312970f2c26880dde0dd51425c8059` |
| `main_source.zip` | `37352b441f2cd69e20566815bd457c0b681115fa52fffbfefee31adc69316596` |
| `supplement_source.zip` | `28ee5dd44865044ccd0f72cef67f6c73096310c2dfa14bd6c1aab921ff5321c1` |

All 58 rendered pages matched clean source-archive builds with
the CVPR style supplied separately. The changed pages were
inspected. PDFs, ZIPs, model weights and data stay local.
[Chinese reproduction](../../../reproducibility/M3W_CURRENT_ZH.md)
