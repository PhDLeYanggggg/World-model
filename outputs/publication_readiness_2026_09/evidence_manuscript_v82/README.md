# Evidence Manuscript v82

The [fixed-slot shape experiment](../donor_shape_choice_v1/README.md)
adds a supplementary section. It compares the learned gain-only and
risk-aware rules with the prefix control and separates the finite-menu
hindsight ceiling from missing easy-reference support. No main claim
is upgraded; the eight-page main stays at
[v81](../evidence_manuscript_v81/README.md).

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v82
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v82.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The 21-page supplement built both from the workspace and from the clean
source archive with the same CVPR style. All 21 rendered pages matched at
85 DPI; pages 20 and 21, including the new table, citations and disclosure,
were visually inspected. The source archive excludes the style and
requires the same CVPR style locally. PDFs and archives remain local.

| Local artifact | SHA-256 |
| --- | --- |
| `supplement/supplement.pdf` | `9af2819dcfedfe49fda812ffe4c063d6e1996157b0f78a437654e4c4db42d80d` |
| `supplement_source.zip` | `327720a67dff0312d94722c477d570a8bfadbfb7d5d25349f038539bbee352a6` |

Development evidence only. Original 2% recording rule, independent
confirmation seal, and submission status are unchanged.
