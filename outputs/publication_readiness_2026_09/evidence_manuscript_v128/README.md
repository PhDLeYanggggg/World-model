# Evidence Manuscript v128

This version adds the recording-allocation study. The main text
reports the dual-reference hindsight limit; the supplement separates
offline timing reallocation, online credit carry, and hindsight
optimization. None of these adds a qualifying calibration policy.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v128
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v128.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v128.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `0b0fcec1780a0bef158ddf5c760812e21009c889d96c3b04e5816c20afd01d28` |
| `supplement/supplement.pdf` (37 pages) | `1352706a5e5a8c23a633f43df5b53cf6030a7147e5851717c3f134ba8a929cad` |
| `main_source.zip` | `f4b68a45033fef2f639107f0fd882e521edb5ec040721a7ebbc415e4f3ea61e1` |
| `supplement_source.zip` | `f4f2052e4ab8322aac37b2afc8ca0c9b22a1a0dc674acfdc6eabe22ab7da1f48` |

All 45 pages match a clean source-archive build at 85 DPI
using the same separately supplied style. The new results were also
inspected on the rendered supplement page. No PDFs, archives, raw
data, or weights are committed. This is a working draft, not a
submission or independent-confirmation claim.

[Allocation and replay](../moving_cv_record_allocation_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
