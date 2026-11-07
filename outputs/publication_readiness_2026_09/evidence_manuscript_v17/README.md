# Decision Manuscript v17

This six-page anonymous working draft adds the
[fixed-recipient neighbor-context ablation](../native_neighbor_motion_adapter_v1/README.md)
to [v16](../evidence_manuscript_v16/README.md). It distinguishes
the relative-path diagnostic from the published joint-minimum metrics,
and reports the negative local-vs-pooled result. Independent evidence
and the submission package remain incomplete.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v17
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v17.py \
  --source-zip "$out/manuscript_source.zip"
cd "$out/build"
tectonic -X compile manuscript.tex
```

The builder verifies the experiment against its score replay receipt.
The anonymous archive excludes official style, data, forecasts and
weights. All six page rasters match after a clean directory rebuild
using the verified official style. Source ZIP SHA-256:
`4533d7ac32600a7dbb14909c469ef83d9241209da33ffa8c4f2afa2c9d564f97`.

[Companion supplement](../evidence_manuscript_v14/supplement_readout.md) |
[Chinese local/CREATE guide](../evidence_manuscript_v12/operation_zh.md).
