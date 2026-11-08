# Working Manuscript v45: Source-Disjoint Calibration

Supplement Section 28 distinguishes missing easy reference support
from a ranking that cannot find feasible actions with available support.
This is a development result. The approved DroneCrowd split, risk rule,
and deployment decision are unchanged.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v45
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v45.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v45.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

The clean source archive builds with the same CVPR style reproduced
all seven main and thirteen supplementary page images. The new
section was visually checked. No enrollment or submission was made.

| Local artifact | SHA-256 |
|---|---|
| Main PDF | `c5aee46b7ecd732b6a63320af21b4859a1a3cef6c6eac165d9ef9015f4b7c188` |
| Supplement PDF | `41c61406c4787ac7303ac9f9623730cbe22f8e0d3390bf8442097206343a8d79` |
| Main source ZIP | `2e750c44c094c65a735f322c491a38df35f2e205ced9a45a79776a37fc28ad09` |
| Supplement source ZIP | `d31838b09d44fc674b16000a8382182cf5639a95ac72fe4e9a745223d1284d3e` |

[Experiment and reproduction](../nested_recipient_calibration_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
