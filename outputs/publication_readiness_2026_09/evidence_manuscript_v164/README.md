# Evidence manuscript v164

This revision adds [source-excluded plan imitation](../source_excluded_slate_v1/README.md).
On OpenDD, the student sometimes recovers plans that pass the
original risk and two-reference checks, but the matched utility
contrast does not establish a joint advantage. On the exposed
camera the student intervenes and still fails. The paper retains
the grayscale plots and keeps the development results separate
from independent confirmation.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v164
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v164.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v164.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `9d917daf3cd601950f80e1ef7c79a9d3c69cc8923cba6843e3743d86bad272d4` |
| `supplement/supplement.pdf` (50 pages) | `603b5e02550ba58581a4bb2e9aaac51d3c35cce593925a687976a84f4821d0a9` |
| `main_source.zip` | `8b867bcaff4eba1ec94a3d83aaf27c998139563cfb3fbf108f5ef17e9ea00b33` |
| `supplement_source.zip` | `dec805c98b3599de01104be9909d2a2b977e80346bd36b5475aa9e2f28ba2552` |

All 58 pages matched clean source-archive renders at 85 DPI,
with the CVPR style supplied separately. Changed pages were
inspected. No conference registration or submission was performed.
PDFs, source ZIPs, caches and weights stay local.
[Chinese reproduction](../../../reproducibility/M3W_CURRENT_ZH.md)
