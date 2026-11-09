# Evidence manuscript v147

The eight-page main paper and the supplement report the OLS-anchored
path-bank comparison and distinguish limits from the placement of
edits from limits that remain even when edits are reallocated.
This is development evidence, not a claim of improved deployment.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v147
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v147.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v147.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `e496905e642afe8eff3e5b0d8da3a75e253d2cb09c2f78e1532ba92d0940e3d9` |
| `supplement/supplement.pdf` (43 pages) | `c7394dd938f903ab37652e20a20b14fbbb200d2030e838492a9a0f5acfc84eec` |
| `main_source.zip` | `186605d95e83eb8c10a52c3e232ca5d554c63b909e3c49e2cf7a620731418fce` |
| `supplement_source.zip` | `389b0072a7e9a2c0cabd2d7247d333c968b7624a2dfb6511280f35fa5232d6e6` |

All 51 pages matched a clean build from the source archives at 85 DPI
using the separately supplied CVPR style. The modified main and
supplement pages were inspected. This check reproduces the document,
not the underlying training or independent validation. PDFs and
archives stay local.

[Experiment](../moving_ols_anchor_menu_v2/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md)
