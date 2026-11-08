# Evidence Manuscript v105

This draft adds [matched support directions](../opendd_support_directions_v1/README.md)
and the frozen-head attribution. Expanded directions add feasible
actions without increasing their radius, but the trained policy
rarely releases on newly feasible views. This is development
evidence, not an independent result.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v105
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v105.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v105.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-support-directions
```

Clean source-archive builds matched all 8 main and 29 supplementary
pages at 85 DPI with the same separately supplied CVPR style.
Main page 7 and supplementary page 29 were visually checked.
The standalone anonymous replay verified 200 count tables and 290
paired intervals across 34 readouts. No confirmation outcomes
were opened and the original risk rule is unchanged.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `e9a496a3e2bcecc243091da7b5ca19bb1887ba412cff14ddd505c246dab9047b` |
| `supplement/supplement.pdf` | `82843d4cb9159634f6227775aa24c1fc890ba6d976006ffcb36a527956af2c21` |
| `manuscript_source.zip` | `c76440cdce9c9bbba1ed13a3e81677e597eb559a7871c6f53d2c4b84413cea81` |
| `supplement_source.zip` | `ec1aeeb2cb82ec80bdb94782eb76055477ddf19d50d672df99899d21f8b85bfa` |
| `opendd_timing_replay_v18/anonymous_opendd_timing.zip` | `52b1f4869a6fa5c7339e7fffff212f8eb8ab8eb1208e4fe02771588e969fd07e` |
