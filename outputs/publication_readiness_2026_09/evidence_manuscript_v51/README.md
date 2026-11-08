# Working Manuscript v51: Finite Easy Harm

Section 34 adds the matched continuation and the exact minimum-risk
menu diagnostic. The [v50 main paper](../evidence_manuscript_v50/README.md)
is unchanged. The new result does not support deployment or independent
confirmation.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v51
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v51.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

The clean archive build reproduced all fourteen page images. The added
section was visually inspected. No enrollment or submission was made.

| Local artifact | SHA-256 |
|---|---|
| Supplement PDF | `73f60ebbfb0f425aa6e79aa8eefef7403f95dba425e72aa42c12c265df50862e` |
| Supplement source ZIP | `efa5cca3727b76643ea07c8d8d06b6ff87feaf2fb88aacb251bd281081014641` |

[Continuation and capacity experiment](../opendd_finite_guard_v1/README.md).
