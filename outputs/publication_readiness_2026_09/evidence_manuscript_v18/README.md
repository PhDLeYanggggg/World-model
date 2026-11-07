# Decision Manuscript v18

This six-page anonymous working draft extends [v17](../evidence_manuscript_v17/README.md)
with the [matched relative-path loss experiment](../native_relative_path_training_v1/README.md).
It distinguishes the observed pairwise improvement from the negative
full-grid lower contrast. There are no independent outcomes or
deployment claims.

The subsequent [completion-aware blend comparison and figure](../native_completion_motion_blend_v1/README.md)
are documented separately and do not change the frozen draft.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v18
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v18.py \
  --source-zip "$out/manuscript_source.zip"
cd "$out/build"
tectonic -X compile manuscript.tex
```

The builder verifies the scalar score replay receipt. The anonymous
archive excludes official style, data, predictions, and weights.
A clean-directory rebuild using the verified style matches all
six page rasters. Source ZIP SHA-256:
`1fae3c09493fb15b924aa2b5c777a7f55c89238146cfbcfd70d216deb60d8db0`.

[Companion supplement](../evidence_manuscript_v14/supplement_readout.md) |
[Chinese local/CREATE guide](../evidence_manuscript_v12/operation_zh.md).
