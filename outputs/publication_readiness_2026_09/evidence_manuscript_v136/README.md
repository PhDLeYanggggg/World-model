# Evidence manuscript v136

The new experiment separates recipient nomination from bounded
path choice and compares the latter to equal-radius shrinkage.
The paper preserves the negative conservative result, despite
observed-step and worst-easy improvements over the weak control.
No model was qualified for independent confirmation.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v136
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v136.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v136.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `ceccdf9429f299a94d8fae2f4dc3e821a728f1e35f831b73dcffe45c00782424` |
| `supplement/supplement.pdf` (40 pages) | `e19a53ba787f89dc89370f9b67de696e4187b710dceeb6b1cc7769b49d4f72b4` |
| `main_source.zip` | `0db263f9904514390e8c516f2d55e25865f2b8046b2029f09663e515f591e9ac` |
| `supplement_source.zip` | `9295cb959f95b93666dabbbf551917bd9e2666b3b34fcf55dc571aba0e44e8b0` |
| `../moving_cv_recipient_path_v1/anonymous_recipient_path.zip` | `ad1ccbfd56e68e560c719d0dfaabd96830c17e43fed8d70a8c1e8a97dd407ca7` |

All 48 rendered pages match clean builds at 85 DPI using the
separately supplied style. The changed pages were inspected.
The NumPy-only archive reproduces both scalar tables after clean
extraction; it does not reproduce training or pair geometry.
PDFs and archives remain local. No submission action was taken.

[Recipient/path experiment](../moving_cv_recipient_path_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
