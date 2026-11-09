# Evidence manuscript v144

The supplement adds the matched causal-prefix experiment, an
arithmetic control without new training, and the separate gain and
risk prediction readouts. The main text is unchanged from
[v143](../evidence_manuscript_v143/README.md).

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v144
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v144.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `supplement/supplement.pdf` (42 pages) | `7362dc8a42042a6d07be7dd16c45ca052c86ea39e17e6490f2eebb0aba1a2741` |
| `supplement_source.zip` | `c664b2687518b48862d5606d781a9e9e666b34dfbd2791114036c4277ec293f7` |

All 42 page renders match a clean source-archive build at 85 DPI
with the separately supplied style. Pages 41 and 42 were inspected.
The PDF, source archive and weights remain local.

[Experiment](../moving_cv_prefix_state_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
