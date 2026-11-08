# Working Manuscript v56: Static Road Context

Section 39 adds the OpenDD static-road comparison and distinguishes the
primary same-count result from the secondary map-state-matched control.
The [v50 main paper](../evidence_manuscript_v50/README.md) is unchanged.
The secondary signal is not an independent result or a deployment pass.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v56
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v56.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

Direct and clean archive builds render identical fifteen-page images.
The new section and references were visually checked. The experiment
was rerun from scratch and the stratified control reproduced exactly.
No enrollment or submission was made.

| Local artifact | SHA-256 |
|---|---|
| Supplement PDF | `f98cc2a9bec79f57fe8e2e91bdbd4e7f12eb30997d4dbaa26bec6ec733b501e0` |
| Supplement source ZIP | `103c8409db1f725257591fe6859b9eb917e15596542e4405defff98d82a0bc41` |

[Experiment and reproduction](../opendd_map_direction_v1/README.md).
