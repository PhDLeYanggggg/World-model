# Evidence Supplement v118

This adds the [OpenDD completion-loss comparison](../opendd_completion_forecast_v1/README.md):
63 matched fits, original risk counts, and a same-peak control that
separates learned path shape from smaller interventions. It does not
change the [main manuscript](../evidence_manuscript_v116/README.md) or
add an independent-confirmation claim.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v118
.venv-pytorch/bin/python scripts/build_m3w_opendd_completion_replay.py
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v118.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

All 33 pages matched the clean source-archive build at 85 DPI with
the same separately supplied CVPR style. Pages 32 and 33 were visually
checked. An extracted anonymous archive independently reproduced
25 descriptive intervals and 12 policy count tables. Nothing was submitted.

| Local artifact | SHA-256 |
| --- | --- |
| `supplement/supplement.pdf` | `c379968e7c6ce20fbccdc8bf385da27a88061888654080adfb8533167c44938c` |
| `supplement_source.zip` | `4a2ffab3d845c96c4449a741886d239b8f89f870350a2958d7d40a6adfd287c5` |
| `opendd_completion_forecast_v1/anonymous_completion.zip` | `b704e65e8366edf0e0b4ca7c9349af4c5803da73f7aee9ee531546dd7beb1d84` |
