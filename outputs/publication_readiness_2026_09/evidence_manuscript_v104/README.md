# Evidence Manuscript v104

This draft adds the [reference-loss guard](../opendd_reference_loss_v1/README.md)
and its frozen-menu thinning bound. Lower reference loss did not yield
a utility-and-risk improvement over the original. The exact matched
capacity is small. No independent confirmation was opened.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v104
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v104.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v104.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-reference-loss
```

Clean source-archive builds matched all 8 main and 29 supplementary
pages at 85 DPI with the same separately supplied CVPR style. Main
page 7 and supplementary pages 28 and 29 were visually checked.
The standalone anonymous replay verified 190 count tables and 271
paired intervals across 32 readouts. These checks are not evidence of
independent performance or submission readiness.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `bf54ff9f3a47bb0a8ea68b43bee225ba4dec62f437e3a6c9c3e60f111123246a` |
| `supplement/supplement.pdf` | `b591e4df291854cce845b721b73b0e82d85995e48e55afc2bf3178ca9092b8de` |
| `manuscript_source.zip` | `a07d4c77bec3d0ab61201e3a92b5193ecfcdf8475ddac78241197bd8526cb7ab` |
| `supplement_source.zip` | `8b4df600cbc612e06b323724e7ab4e054eea1d3d6cfb26277f9b03b4c13f458a` |
| `opendd_timing_replay_v17/anonymous_opendd_timing.zip` | `8a32526286d3319385886d177f354eea028449fc35d7b8bcc00bf4df6c51c37a` |
