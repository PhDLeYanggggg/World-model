# Working Manuscript v47: Optional Query Allocation

Supplement Section 30 records the matched risk-utility tradeoff
and the exact OLS4-only ceiling. The [v45 main paper](../evidence_manuscript_v45/README.md)
is unchanged.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v47
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v47.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

The clean archive build reproduced all thirteen page images. Section 30
was visually checked. No independent confirmation, enrollment or
submission was made.

| Local artifact | SHA-256 |
|---|---|
| Supplement PDF | `739d5df4d7d46d3e1e6729dc08e762d57787dc26157c4df6d048eef32ffc0a19` |
| Supplement source ZIP | `5db7f91d1b192879755fd0bfd7017ae8819e4a5194edb10b2454ab591f2b9ead` |

[Experiment and reproduction](../optional_allocation_v2/README.md).
