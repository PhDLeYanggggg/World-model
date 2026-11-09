# Evidence manuscript v142

The supplement adds a same-bank comparison of prefix proposals,
risk-budgeted query options, and unrestricted per-agent hindsight
decisions. It also reports the frozen scorer at matched counts.
The main text is unchanged from [v141](../evidence_manuscript_v141/README.md).

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v142
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v142.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `supplement/supplement.pdf` (42 pages) | `e8a6ffbd5ee3a400596230b3410eee10505396cbdfa82075b5a5ed6d3b969094` |
| `supplement_source.zip` | `e9866cc414b322c1e9e24de64cca4ff985df6f29212d2df2c994e78c80500891` |

All 42 page renders match a clean source-archive build at 85 DPI
with the separately supplied style. Pages 41 and 42 were inspected.
The PDF and source archive remain local. These are development and
hindsight results, not independent confirmation or a submission.

[Experiment](../moving_cv_risk_frontier_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
