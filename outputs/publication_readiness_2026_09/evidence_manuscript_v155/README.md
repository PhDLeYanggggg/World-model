# Evidence manuscript v155

The manuscript records a matched initial-forecast experiment on Moving
development sources. The supplement separates visible-label accuracy
from missing-step-debited utility, and an unmatched calibration screen
from a comparison at identical intervention times, recipients and
displacement radii. Neither trained loss adds a passing view at fixed
budget. The original risk rule, deployment and independent
confirmation status are unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v155
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v155.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v155.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `26d39333c4efd35b80cac9c5172d6539f265ac9ae695da4b167ddd8ccfd244d3` |
| `supplement/supplement.pdf` (47 pages) | `0bf47c173401a28c909cb7bfbd8f8813190634b23836aaab70dd657f121f15cb` |
| `main_source.zip` | `063f4e3fc04514fa97f7a8c7d019fc1d2ccdb824c6a2cd1f71b5f045e88f2046` |
| `supplement_source.zip` | `514aa99771370abc7e1b56b491b783809c4dd3a3699cfec01d4cf9b961ef8705` |

All 55 pages match clean source-archive builds at 85 DPI with the
separately supplied CVPR style. The changed manuscript and supplement
pages were visually checked. PDF and ZIP copies remain local. This is
a working research draft, not a submission or independent confirmation.

[Experiment](../moving_dual_record_forecast_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md)
