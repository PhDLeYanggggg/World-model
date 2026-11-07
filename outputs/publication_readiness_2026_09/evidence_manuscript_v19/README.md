# Decision Manuscript v19

This six-page anonymous working draft extends [v18](../evidence_manuscript_v18/README.md)
with the [continuous pair-blend directional audit](../native_pair_blend_direction_v1/README.md).
It distinguishes verified query-local improvement from original
recording risk and independent evidence. It is not ready to submit.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v19
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v19.py \
  --source-zip "$out/manuscript_source.zip"
cd "$out/build"
tectonic -X compile manuscript.tex
```

The builder checks the result against its replay receipt. The
anonymous source archive excludes the official style, data, weights
and forecasts. A clean-directory rebuild with the verified style
matches all six page rasters. Source ZIP SHA-256:
`7a815fee5dea9d5f5b50b00282633908f0c0b3972192f7a77feee7d8cff4c069`.

[Companion supplement](../evidence_manuscript_v14/supplement_readout.md) |
[Chinese local/CREATE guide](../evidence_manuscript_v12/operation_zh.md).
