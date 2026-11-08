# Working Paper v66: Feasibility and Causal Discrimination

The main paper now foregrounds what the experiments establish: a forecast
menu can admit useful actions under a recording risk constraint without
a learned controller being able to identify them. The abstract, OpenDD
results and two-panel vector figure now include the verified capacity
and matched peer-context comparison. This is an anonymous working
paper, not a submission-ready result or a generalization claim.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v66
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v66.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v66.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The main draft has eight pages; the supplement has eighteen. Clean
source archive builds reproduce all page renders. The abstract, new
results and figure, and bibliography endings were checked visually.
PDF digests identify the verified local files, not byte-identical
recompiles. No outcome from the 30 sealed confirmation groups is used.
Nothing was enrolled or submitted. Local PDF and ZIP files stay out of Git.

| Local artifact | SHA-256 |
| --- | --- |
| Main PDF | `04f7e2b0670760f3d48a4a5c5b676e25ab6448d2173f0643e058d3b2855263a5` |
| Main source ZIP | `61fe50e6f3310dad65339dbcba2dbf390ab4a9b15d2b91e8be3d5a84dddb104a` |
| Supplement PDF | `0957f73443b5e7d4e12813a57eeb1ee2aeff452ba6eebee70f99f998fc98ed69` |
| Supplement source ZIP | `39f7a525c78b849261ae295d740473078943d4620c12e7162c24cb95d1e0bdcc` |

[Direct action experiment](../direct_count_value_v1/README.md) |
[Chinese local/CREATE runbook](../local_create_runbook_zh.md).
