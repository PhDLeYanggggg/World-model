# Working Manuscript v42: Allocation Capacity and Actual Risk

Section 8 and Supplement Section 25 now distinguish the value of
changing recipients, redistributing the same total, and freeing
counts. The leave-source-out predictor fails the unchanged risk
rule even where there is hindsight capacity.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v42
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v42.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v42.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

Clean source-archive builds with the same official style reproduced
all seven main and twelve supplementary page images. The changed
passage and table were inspected. The inherited line-number encoding
warning remains. No enrollment or submission was made.

| Local artifact | SHA-256 |
|---|---|
| Main PDF | `63d45162e37601a16d40933c0f5d97bf28c54c562c4bc7e4c327bddc2a3d796f` |
| Supplement PDF | `5fdc9f1639132e3a8b097f76563732b3ec765eb622e1606a0f4823e843adc49e` |
| Main source ZIP | `a3d9e5ec92a7d3647739eacd41f12fc046a9739b1d3eb64c8bb021c7d9813146` |
| Supplement source ZIP | `fcab36e2430f8448d3738d903602b858baa258ba45d7413a00a3269d053ecc7d` |

[Experiment and replay](../recipient_release_capacity_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
