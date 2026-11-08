# Evidence Manuscript v83

The supplement adds the [compact EqMotion producer comparison](../opendd_comoving_eqmotion_v1/README.md).
It separates whole-roster trajectory quality, matched-radius intervention
outcomes, and hindsight capacity. The eight-page main remains at
[v81](../evidence_manuscript_v81/README.md); no main claim is upgraded.
The recent matched OpenDD results have a separate
[anonymous scalar replay](../opendd_candidate_replay_v1/README.md).

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v83
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v83.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The 22-page supplement built from the workspace and from a clean
extraction with the same CVPR style. All 22 rendered pages matched at
85 DPI; the changed experiment page, disclosure and reference spillover
were visually inspected. PDFs and archives remain local.

| Local artifact | SHA-256 |
| --- | --- |
| `supplement/supplement.pdf` | `021e1d69cf1007ace5341d72f6395957eae44133a8f1bf2fbc308bca39e6ddbf` |
| `supplement_source.zip` | `7a251de47d8e0065fd6984de29d15ce89e502066f75f0fcf7bbf997135065f14` |

Development evidence only. Original 2% recording rule and the
independent confirmation seal remain unchanged. No submission.
