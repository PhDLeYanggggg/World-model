# Working Manuscript v38: Broad Motion and Predicted Risk

The main calibration section and Supplement Section 21 separate three
findings: the learned gain rule trails full OLS4, the finite price search
misses a nonempty predicted feasible set, and the search repair still
loses to causal hindcast at the same per-query blend strengths. These are
diagnostics on opened calibration recordings. No deployment policy was
selected and no independent confirmation was opened.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v38
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v38.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v38.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

Clean extraction with the same official `cvpr.sty` reproduces all seven
main and eleven supplementary page renders. The changed main page and
new supplement table were inspected. Extracted text was screened for
local paths and author names. This checks the source build, not a cold
training replay or a final submission audit.

| Local artifact | SHA-256 |
|---|---|
| Main PDF | `21e35bc28d051ce8cf68696df8cc59ca0824b50a371ba59f37fd12b1c4d759c0` |
| Supplement PDF | `6b72eaf55e0db9437254553bd9cb923e37819afc0e7c0378fa2f9dba65cbdbe2` |
| Main source ZIP | `1a14a215ff89a62c4c3cdfc08c482eb57e268c98db19aa54a27aa8420d6dd5bb` |
| Supplement source ZIP | `782ef102ff2b5c8d75958a10e77f591e3ba7c1256dff1170cbb7f74a04674970` |

[Experiment and replay](../dense_motion_calibration_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
