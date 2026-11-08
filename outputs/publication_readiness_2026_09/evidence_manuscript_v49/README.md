# Working Manuscript v49: Released Label Support

Section 32 adds exact raw-label reconciliation and the matched
OpenDD refit. The [v45 main manuscript](../evidence_manuscript_v45/README.md)
is unchanged. The new section reports exposed development only.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v49
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v49.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

The clean source archive reproduced all fourteen page images. The
added section and references were visually checked. No enrollment
or submission.

| Local artifact | SHA-256 |
|---|---|
| Supplement PDF | `bd6fc6657cd974796a3c79492cb8c03585b8a1ad845a9f14df7385b9e1e13f19` |
| Supplement source ZIP | `7e1358ead8f5e19aede97c524670f7785ae9e02bf52c00c3e77b8f6c45adfdb7` |
