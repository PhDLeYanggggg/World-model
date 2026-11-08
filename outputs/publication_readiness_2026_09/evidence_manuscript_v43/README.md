# Working Manuscript v43: Matched Risk and Recipient Feasibility

Section 8 keeps the main result concise. Supplement Section 26 gives
the 54-head, three-seed matched test, the risk-component swap and
the exact hindsight path-only feasibility check. None qualifies
as deployment or independent confirmation.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v43
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v43.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v43.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

Clean builds of the two archives with the same CVPR style reproduce
all seven main and twelve supplementary page images. The new passage
and table were inspected. The inherited line-number encoding warning
remains. No enrollment or submission was made.

| Local artifact | SHA-256 |
|---|---|
| Main PDF | `8b3859cb4939f77e2c0cefacf144c2d5e6e49bdf0cf0666ca555679512310c7b` |
| Supplement PDF | `276ad13d358f0bb1bc4d7e635c8b34cc16c5da1b1c13f7c788344c2d738a38c8` |
| Main source ZIP | `ac16f8669670414f7de5a17e26d019ac69d700cb6f06a18838aaef98ea10436e` |
| Supplement source ZIP | `462a00131d8997e769e904bbc68f696911cf2af5421416df1cd6972fc55c2318` |

[Experiment and replay](../matched_support_moments_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
