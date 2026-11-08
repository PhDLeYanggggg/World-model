# Evidence Manuscript v85

The supplement adds the
[double-excluded candidate-advantage study](../opendd_candidate_advantage_v1/README.md).
It distinguishes actual equal-count causal performance, the
accuracy-only oracle, and the original-risk assignment ceiling.
The [v81 main](../evidence_manuscript_v81/README.md) is unchanged.
The [extended anonymous replay](../opendd_candidate_replay_v2/README.md)
covers eight development readouts without private paths or data.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v85
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v85.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The 23-page supplement builds both locally and from a clean extraction
using the same CVPR style. All 23 rendered pages matched at 85 DPI;
pages 22 and 23, including the new table, were visually checked.
PDFs and archives remain local.

| Local artifact | SHA-256 |
| --- | --- |
| `supplement/supplement.pdf` | `c67db846e802d291e7022989b103aa8354d2cc51cc063695211f0ab2c1d286ac` |
| `supplement_source.zip` | `6b9c4ec06a941f931d4cc1a829a8c97a2292ac09f72890c767c9aa582b1ae30e` |

Development evidence only. No policy promotion, original risk change,
independent confirmation or submission.
