# Working Manuscript v41: Past Sampling and Risk Capacity

Section 8 and Supplement Section 24 add the matched-anchor sampling
contrast and the optimistic convex-hull risk bound. The bound is a
fitting-only hindsight exclusion for one candidate family; it is not
an inference method or a global impossibility claim.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v41
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v41.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v41.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

Clean archive builds with the same CVPR style reproduced all seven
main and twelve supplement page images. The changed passage and
table were inspected. The inherited line-number encoding warning
remains. No enrollment or submission was made.

| Local artifact | SHA-256 |
|---|---|
| Main PDF | `eebef30c2d81d9a45b7a6e75636a24c14bd3e04ef9232a88c6ced74e1eb1cbdd` |
| Supplement PDF | `368ad371e6dc8ca8731358a7abd41a8775e54e458e8f22db879aac24980427a2` |
| Main source ZIP | `0f75ffb692a49025cc656816e8295bfb5e8f05820fed21d4102c7952e404b2c0` |
| Supplement source ZIP | `58bfc5a6f7449c13b552a85f8a8dd1836f67e4b55c4294f275463870617c7dad` |

[Experiment and replay](../past_sampling_audit_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
