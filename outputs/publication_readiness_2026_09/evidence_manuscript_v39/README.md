# Working Manuscript v39: Standalone Hindcast Control

Section 8 and Supplement Section 22 add a matched-recipient comparison
for the fitting-selected causal hindcast. The manuscript gives the
40/40 opened-recording pass alongside the 0/24 fitting-domain result,
without implying a deployment or independent confirmation.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v39
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v39.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v39.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

With the same official CVPR style, the two source archives reproduced
all seven main and eleven supplement page images in clean builds.
The updated main passage and supplement table were visually inspected.
The inherited line-number encoding warning remains. This is a working
manuscript build check, not an independent experiment or a submission.

| Local artifact | SHA-256 |
|---|---|
| Main PDF | `2b68924e9ca10ac8894c4fba7a045c8d77287b526f4830a83a79e45c96186286` |
| Supplement PDF | `c508572b7b394c597a081d49666ee3b86c1a92d3b88541bbdfd8ffb3fad64959` |
| Main source ZIP | `5bfe269fbb7230ea0023b9d43ad224b4102d6bbd1dce8b5e832d9de6f1bb1a50` |
| Supplement source ZIP | `ce7c6757089a68aa04efd932ce526b7cf0ff081acbfb0a9588d0893cfb9dde0a` |

[Experiment and replay](../standalone_hindcast_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
