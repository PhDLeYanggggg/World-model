# Working Manuscript v37: Intervention Capacity

The main calibration section now gives the perfect-switch bound and
the unrestricted outcome-aware witness. Supplement Section 20 derives
the bound and gives Table 17 with 15-source-group intervals. These
are calibration diagnostics, not new training, a change in the 2%
rule, or a deployable policy.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v37
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v37.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v37.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

The anonymous source archives require the official `cvpr.sty`. With
that same style, clean extraction and compilation reproduce all
seven main and ten supplementary page renders exactly. The changed
page and new table were visually inspected; the inherited line-number
encoding warning remains. Extracted text has no workspace path or
author-name token. This is a build check, not a cold training replay.

| Local artifact | SHA-256 |
|---|---|
| Main PDF | `9c8988863124db8c47ceb81853eda9566b19097e6cf291f4201d21fca374082c` |
| Supplement PDF | `a051a4697f7e60d25cb2266667378aeb549615bdb4628fb377b3212dfe3dc649` |
| Main source ZIP | `c46571a029ceaef5a74d3cee7af581aa2cbe27fb70ed84873abd9332bc4d3d30` |
| Supplement source ZIP | `71eccf4c3b4635977cf05a32061a040c3af982f059fb00e2f43c8b3b99bf2004` |

[Calibration capacity experiment](../dronecrowd_perfect_switch_ceiling_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
No registration, submission, or independent confirmation was performed.
