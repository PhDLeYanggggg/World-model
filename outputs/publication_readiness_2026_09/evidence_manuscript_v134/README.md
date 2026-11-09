# Evidence manuscript v134

The main text separates the full bank at frozen counts from the
full bank at free counts. The supplement reports the four-arm
comparison, nested witness verification, and source bootstrap.
This is development evidence, not independent confirmation.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v134
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v134.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v134.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `aed89098c61b2262efe1868865859a9d4f1be53429c1653a0542900693d28ea7` |
| `supplement/supplement.pdf` (39 pages) | `4025398092bca897f8a0f73c44901e53f9d403011ef95b11b4bb35d1078de886` |
| `main_source.zip` | `e73e22c46127e254353d42079240733bd96ce819c1578e5b0c1789a514301f27` |
| `supplement_source.zip` | `f8930be9035dbf2fe63933802e4c74995cb32b92f732f7e4d5d2b049dcb822f7` |
| `../moving_cv_budget_menu_factorial_v1/anonymous_capacity.zip` | `0e3d347a43d8f90280c17544a1691db02ebcb36f4703a22dd1643096dd727635` |

All 47 rendered pages match clean source-archive builds at 85 DPI
with the separately supplied style. The changed main page and
supplement table were inspected. The anonymous scalar archive
was verified from a clean extraction. The PDFs and ZIPs remain
local. No submission or deployment action was taken.

[Factorial experiment](../moving_cv_budget_menu_factorial_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
