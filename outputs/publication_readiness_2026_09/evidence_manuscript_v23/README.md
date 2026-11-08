# Decision manuscript and supplement v23

The six-page main draft and five-page supplement add the [paired synthetic-gap check](../native_completion_occlusion_v1/README.md) and [motion-centered sign witnesses](../native_motion_centered_completion_v1/README.md). Both analyses bind to exact replay receipts. The new center improves pseudo-gap coverage; neither it nor a positive reference world identifies the true missing-label ordering. The official scores, recording rule and independent evaluation roles are unchanged. This is a working draft, not a submission-ready result.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v23
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v23.py main \
  --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v23.py supplement \
  --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The anonymous source archives exclude the official style, raw data, weights and trajectories. Clean-directory builds using the verified CVPR style were pixel-identical on all rendered pages. The new text was checked in both PDFs. The source archive SHA-256 checksums are:

- Main: `711a73b1bb5ed7ac1d70f26945f0b1ffd8a2bfccc1fa2c09df57bac5b53426ae`
- Supplement: `0f0a47df8aec55cd0be812003379b1f680e8d37fc5b0482befd2c212cbba20e5`

[Aggregate replay](../review_replay_v1/README.md) | [Chinese local/CREATE guide](../evidence_manuscript_v12/operation_zh.md)
