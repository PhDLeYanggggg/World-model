# Evidence Supplement v120

This adds the [past-backtest expert mixture](../moving_prefix_expert_mixture_v1/README.md):
matched model and edit-size controls, and a complete-label diagnostic
showing that expert ordering and selected forecast cost diverge at
longer leads. The [v119 main paper](../evidence_manuscript_v119/README.md)
is unchanged. No new independent confirmation or submission.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v120
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v120.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

All 34 pages matched the clean source-archive build at 85 DPI
with the same separately supplied CVPR style. Pages 33 and 34
were visually checked. The extracted anonymous replay recomputed
17 contrast tables (51 interval entries) on the 102 repeated views.

| Local artifact | SHA-256 |
| --- | --- |
| `supplement/supplement.pdf` | `96ff798b2594c9d97277c4bef358cf4a6acaef255ec7e183dba97495102d5b9e` |
| `supplement_source.zip` | `57ddeaa09c642f016b92b246d477e56645acbb1b2c58c1c1e73f1a3026c4f6aa` |
| `moving_prefix_expert_mixture_v1/anonymous_prefix_experts.zip` | `1633091419d9202cc3d7c7ea4319c65e82e64ddeeb604f7ed2fc45b8b11e4f5d` |
