# Evidence Supplement v121

This adds the [matched cost-target experiment](../moving_motion_choice_cost_v1/README.md).
The new section separates the observed improvement in decision cost from
the inconclusive full-grid bounds and the failed 2% risk checks. The
[v119 main paper](../evidence_manuscript_v119/README.md) is unchanged.
These are working materials, not a submission or independent confirmation.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v121
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v121.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

All 34 pages matched a clean source-archive build at 85 DPI with the same
separately supplied CVPR style. The added section and table on pages 33-34
were visually checked. The data, checkpoints and PDFs are not in Git.

| Local artifact | SHA-256 |
|---|---|
| `supplement/supplement.pdf` | `f0b19e26a00c0754045b641e97ef191b13364f1d04770009d504378d8f8a005b` |
| `supplement_source.zip` | `f69d3c813ea2ca7253e18d57661f59723b252dccfdad7ac37ce83da8e952575b` |
