# Working Manuscript v53: Matched Path Choice

Section 36 adds the OpenDD-trained path chooser and its action-only
ablation. The [v50 main paper](../evidence_manuscript_v50/README.md)
is unchanged. This supplement distinguishes the positive hindsight
witness from the negative result of learning to choose from the past.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v53
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v53.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

Direct and clean archive builds render the same fifteen pages. The new
section and final reference page were visually checked. The selector
evaluation replayed exactly in a clean scratch run without training.
No enrollment, submission or independent confirmation was performed.

| Local artifact | SHA-256 |
|---|---|
| Supplement PDF | `22f9e27d6f4edbdf80164d75e266a5191513008d3e78736895c9827572f5c51b` |
| Supplement source ZIP | `f44d1a6c37a8a190ccdcedc579e6ee5ad5efc28aace470b353b20b9fcb3594af` |

[Experiment and reproduction](../matched_path_choice_v1/README.md).
