# Evidence Manuscript v84

The supplement adds the [hard/easy contribution audit](../opendd_component_guard_v1/README.md)
and [same-budget difficulty test](../opendd_hardness_gate_v1/README.md).
They distinguish the raw forecast's easy/hard trade-off, the effect of
removing directional components at equal displacement, and the causal
classifier's performance at equal counts. The
[v81 main](../evidence_manuscript_v81/README.md) is unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v84
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v84.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The 22-page supplement builds from the workspace and from the clean
source archive with the same CVPR style. All 22 rendered pages agree at
85 DPI; the new experiment, table, disclosure and references on pages
21 and 22 were visually checked. PDFs and archives remain local.

| Local artifact | SHA-256 |
| --- | --- |
| `supplement/supplement.pdf` | `3a498933210849990c51aec2fc588ceb3b6742ba2c8c816d40c584526be23923` |
| `supplement_source.zip` | `a3ae3e4641c45ad7139955d512df067f1886bc4fb6414485875176ed042f0cf4` |

Development evidence only. No promotion, independent confirmation or
submission. The original risk and support rules remain unchanged.
