# Evidence Manuscript v129

The main text now states the perfect-edit coverage bound on the held
fitting recordings. The supplement adds the matched lead-mixture
training comparison and distinguishes feasible finite-menu witnesses
from an optimistic unrestricted upper bound. No calibration qualified.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v129
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v129.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v129.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `343b03be88c9daaa209624ebd539d04c966920def504d5ce35b90c51a00fd500` |
| `supplement/supplement.pdf` (37 pages) | `718236aad7eae4e215e9a1b83537716a222028caf9e818e4366e2962717f3816` |
| `main_source.zip` | `2dd442886050fcf4fc8c01612a5a3fb732162937a41b0099b81a3d03873602d7` |
| `supplement_source.zip` | `9d2a4cc742e84f24b11b2aeee6368386b752460c191fd6a51f81cdf4f3287552` |

All 45 pages match a clean source-archive build at 85 DPI using the
same separately supplied style. The new supplement section was also
visually checked. No PDFs, archives, raw data, or weights are
committed. This is a working draft, not a submitted paper.

[Experiment and necessary coverage bound](../moving_cv_lead_mixture_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
