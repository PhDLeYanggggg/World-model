# Working Manuscript v59: Optional Reference Bridges

Section 42 adds the original, clipped and unrestricted hindsight
comparison. The [v50 main paper](../evidence_manuscript_v50/README.md)
is unchanged. The broader menu expands possible interventions, not
measured deployment performance.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v59
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v59.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

Direct and clean archive builds render identical sixteen-page images.
The new section and references were visually checked. The optional and
clipped solves replayed exactly. No enrollment or submission was made.

| Local artifact | SHA-256 |
|---|---|
| Supplement PDF | `ceb461c4b17bcbf0a1de82fd7415f8e676d408df2b9a8da6f4a2447eb0eac2d1` |
| Supplement source ZIP | `935a355ff557c08414a71bd857f3af3db35432a20123d31f96a4f124d87e7620` |

[Experiment and reproduction](../optional_reference_bridge_v1/README.md).
