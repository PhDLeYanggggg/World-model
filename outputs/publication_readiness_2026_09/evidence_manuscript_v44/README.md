# Working Manuscript v44: Recipient Feasibility and Query Fallback

The main paper and Supplement Section 27 record the shared-path
recipient test, the exact predicted easy-risk capacity check, and
the query-local fallback. The recording-budget test is labeled
offline. There is no new deployment or independent confirmation.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v44
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v44.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v44.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

Clean source archive builds with the same official style reproduce
all seven main and thirteen supplementary page images. The new
section was inspected and the source was checked for identity
tokens. The inherited line-number encoding warning remains. No
enrollment or submission was made.

| Local artifact | SHA-256 |
|---|---|
| Main PDF | `cba073596baa1bb1271417c9a2145e519e67bd4c6a367f71cd85426f8a7e0a55` |
| Supplement PDF | `5b82a90eefe27520caedebdd1ba8c98fe9ebb1d158a0d0fb57a235b3d2535ee8` |
| Main source ZIP | `d406ecdadf829062651d9856da931f7fd1dc82e7ac39b70ae238d0ec47fbe1b9` |
| Supplement source ZIP | `17ed6d2648c0408f2c0d31bc269f6757ee64d521416d416c9c514ab3aa8438f9` |

[Recipient experiment](../recipient_feasibility_v1/README.md) |
[Query-local fallback](../recipient_query_fallback_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
