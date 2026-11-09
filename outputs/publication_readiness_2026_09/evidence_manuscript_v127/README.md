# Evidence Manuscript v127

This version adds two controlled fitting-source comparisons: screening
order with frozen scores, and conservative within-agent preference loss
with matched retraining. Neither produces a qualifying calibration
policy. The supplement records the known easy tail reduction separately
from unsupported and excessive-harm recording views.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v127
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v127.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v127.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `ccfc5815481124a112c36a398827302a3bf0c027cac4ee378180705b0a74d6d6` |
| `supplement/supplement.pdf` (36 pages) | `dd8d23bfaf43e0c20217563be5c4beb058bf0732cca5171c4de76d355cc49bc9` |
| `main_source.zip` | `8e1fc8c2991014c267338244f17ac874047473089399a0a570e3ec6ad99f3c09` |
| `supplement_source.zip` | `0732d4a8417f83bca2880f9e904564218902e35f3d4b45833b23979d890aceec` |

All 44 rendered pages match a clean source-archive build at 85 DPI
using the same separately supplied style. The new results page was
also inspected visually. No PDFs, archives, raw data, or weights are
committed. This remains a working draft, not a submitted paper or an
independent-confirmation claim.

[Screening order](../moving_cv_screen_order_v1/README.md) |
[Preference experiment](../moving_cv_paired_nomination_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
