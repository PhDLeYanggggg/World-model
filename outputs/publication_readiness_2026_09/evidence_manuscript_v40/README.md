# Working Manuscript v40: Norm-Matched Shared Correction

Section 8 and Supplement Section 23 report the causal-input,
same-recipient, same-norm contrast. The new comparison does not
attribute the transfer gap to camera motion, and does not establish
deployment or independent confirmation.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v40
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v40.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v40.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

Clean archive builds with the same CVPR style reproduced all seven
main and eleven supplement page images. The changed passage and table
were visually checked. The inherited line-number encoding warning
remains. Neither an enrollment nor a paper submission was made.

| Local artifact | SHA-256 |
|---|---|
| Main PDF | `7885220c41f62903ec48604d009226993e03405b6c68aff5dae7cb139f462866` |
| Supplement PDF | `74fd607af814a116e31b205aa0a480d0387a6d2875b6160cc6b72181d56ef398` |
| Main source ZIP | `58233c4ccbb3830749a4cbfffedf35a5272ee5be580b7f2ef73eb11f18d5ab67` |
| Supplement source ZIP | `b7ebcd8fe530acbb50307a60bde1c2a38c6567865b625f941600f7857eb37aa2` |

[Experiment and replay](../shared_correction_audit_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
