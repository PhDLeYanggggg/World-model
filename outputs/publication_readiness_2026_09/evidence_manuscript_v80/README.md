# Evidence Manuscript v80

Integrates the [first-edit training and timing ceiling](../first_edit_target_v1/README.md).
The main text separates the shared-head event improvement from the
stronger frozen control, then distinguishes first-query capacity from
hindsight timing capacity at the same one-edit budget.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v80
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v80.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v80.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The 8-page main and 21-page supplement built from the working files
and fresh source archives using the same official CVPR style. All 29
page renders matched at 85 DPI; the changed pages, disclosures and
references were visually checked. PDFs and archives remain local.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `c8a8bb1f1e08f049bcdedb7e8a76075851066a9aae4c3276bbf3dc772d1fa51f` |
| `manuscript_source.zip` | `7d2103bad01c32f5a5c008db1adee03c191e247256c862c671d1fefa73d35ec0` |
| `supplement/supplement.pdf` | `4f3e0da8218ea432b4c7672c2cd671e46c8bc04b0304806df83223027b1072ed` |
| `supplement_source.zip` | `0d7a7b4eed0f069045fe0c2b43a1e4c4f41f29811b90dafe0780d75fe2550473` |

Development evidence only. No independent confirmation, submission or
claim of readiness.
