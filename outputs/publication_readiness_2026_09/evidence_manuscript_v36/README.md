# Working Manuscript v36: Seed-Shared Completion Sensitivity

The seven-page main and ten-page supplement now include the frozen
OpenDD context sensitivity. The main qualifies both direct context
comparisons. Supplement Section 19 derives the cross-seed bound and
Table 16 reports the observed term, outer interval and attained
template values separately. No model, 2% risk result, or confirmation
role changed. The paper remains a working draft, not submission-ready.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v36
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v36.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v36.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

The anonymous source ZIPs require the official `cvpr.sty`. Both were
extracted, supplied the same style, and compiled in clean directories.
All 17 rendered pages matched pixel for pixel. The changed main page,
equation, table and disclosure were visually checked; the inherited
line-number encoding warning remains. The ZIP member allowlists contain
only draft sources and vector figures. Extracted PDF text was checked
for workspace paths and author-name tokens. This verifies the draft
build, not cold model training.

| Local artifact | SHA-256 |
|---|---|
| Main PDF | `e2eecced8548a1fad2418ff4c51cab82737ce83ace2b1052c6b23a193e172ed0` |
| Supplement PDF | `9b7f011a806475787788a7ed97837e7433414688eae6237da33d4cff3025e277` |
| Main source ZIP | `398d6b6ad60f8eadaa08390bce34b46256dc607e762fddf7384301542b1019f6` |
| Supplement source ZIP | `8ead557561eed69b4766794b76e3f1c8ca0c9f6c566c7bd607177c3e10c22843` |

[Underlying experiment](../opendd_seed_shared_completion_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
The [official CVPR 2027 calendar](https://cvpr.thecvf.com/Conferences/2027/Dates)
was rechecked: registration November 10, main November 16, supplement
November 23, all AoE. No registration or submission was performed.
