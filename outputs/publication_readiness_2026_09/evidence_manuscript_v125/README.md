# Evidence Manuscript v125

The main paper now distinguishes the feasible hindsight combinations from
what the causal pair head actually recovers. The supplement adds the
source-excluded donor fits, exact-count choice comparison, matched query
displacement check and unchanged risk readout. These are exposed development
results. Independent confirmation and submission are not complete.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v125
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v125.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v125.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `afa7bbea7197e8198e0f00bbc83377dd373d30326e207632323e4524a1920583` |
| `supplement/supplement.pdf` (36 pages) | `170173da57a77d9645e8b05029212ac3d1f9ad3080d981ed873c8d52371874b4` |
| `main_source.zip` | `0647a8882397fac4049f65e574766e96ee73fb017bbaa83e5c5bb44c3f019201` |
| `supplement_source.zip` | `6b1ce47d435443371e1bade3ed518776b6997c72c7b30b0bb61053e03c01b009` |

[Experiment and anonymous replay](../moving_cv_joint_selector_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
All 44 pages matched clean source-archive builds at 85 DPI with the same
separately supplied CVPR style. The new result paragraphs were visually checked.
No PDF, raw data, or private model is committed. Not submitted.
