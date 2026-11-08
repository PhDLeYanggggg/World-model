# Evidence Manuscript v92

I added the [fixed-shape and sparse-policy results](../opendd_reserve_shape_v1/README.md).
The main paper reports the sparse policy's gain over CV on exposed
OpenDD, together with its failed recording-risk gate. The supplement
includes both directional shape contrasts and the unchanged full-roster
counts. It does not claim independent confirmation.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v92
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v92.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v92.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-reserve-shape
```

All 8 main and 25 supplementary pages match clean archive builds
at 85 DPI, with the same separately supplied CVPR style. Main page
7 and supplement page 25 were visually checked. The anonymous archive
reproduces 74 count tables and 89 intervals from scalar evidence across
11 readouts, not upstream training. PDFs and archives remain local.
No submission was made.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `b7247825b6fbe4400b0a261aa3b3e12d25c52e8fedba323b522a364d47e25707` |
| `supplement/supplement.pdf` | `95f705c9b45968d82e283cdec6c6fd68d36bab7b249df5e29144bb240a0a2afa` |
| `manuscript_source.zip` | `0020a0f2a3e3ad01576fac18c4a26043a65419c4167f7045f5466e320b08e729` |
| `supplement_source.zip` | `611325f8d6151bc5ae4c7143b70605c061cfcb1734884acd96b6e24354c4c29d` |
| `opendd_timing_replay_v5/anonymous_opendd_timing.zip` | `c9bd2592d3d8b6406af8af687c0e649c5f9aff07abfeccbd4cb5ddae2011c3dd` |
