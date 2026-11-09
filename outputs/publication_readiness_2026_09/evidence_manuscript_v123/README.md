# Evidence Manuscript v123

This version contrasts two starting forecasts. OLS4 cannot be made safe by
one arbitrary edit per query on the exposed roster. Starting at CV avoids
that specific problem, but its one-edit analytic menu cannot certify gain
over OLS4 under even relaxed risk constraints. The main paper reports both
limits; the supplement gives the top-k and dual-bound derivations and the
capacity table. This is development evidence, not independent confirmation.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v123
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v123.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v123.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

All 43 pages matched the clean source-archive builds at 85 DPI with the
same separately supplied CVPR style. Main pages 7-8 and supplement pages
34-35 were visually checked. These are unsubmitted working materials.

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` | `e769b7e5db363d80931ab6492e463fa109f2072713a4087f0d6a9981a4fb5396` |
| `supplement/supplement.pdf` | `f63efda04be442caca5911522438d3e1e84cf9cd908289ffa06d0f4250de5ab7` |
| `main_source.zip` | `cf64aee39deb43d4a1769f52a35b16e666b9d558e2dc397d9a91317960b57180` |
| `supplement_source.zip` | `a57ba8760a45af4eb3211972935344738dcb648e02e94292e1af1851accc0fa0` |

[Capacity results and plot recipe](../moving_cv_anchor_budget_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md)
