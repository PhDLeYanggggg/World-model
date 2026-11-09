# Evidence manuscript v152

The main results now separate an improvement to the candidate bank
from a gain in causal assignment, replacing a dense catalogue of
older exploratory experiments. The supplement retains those studies
and adds the matched teacher comparison and fixed-recipient capacity
diagnostic. The document remains a development draft.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v152
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v152.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v152.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `dc7bfcd201f303882e876d4517a243c78bec81f6bea4d2493183ac522c8bcf31` |
| `supplement/supplement.pdf` (45 pages) | `63ac110efdacc7c17c597047e651dbba749e415061a0f77cacdfb80496780c65` |
| `main_source.zip` | `c41b85d4e930c9da72296e727d4db937c17b6f9b4043c55ab5710a6081fcde9c` |
| `supplement_source.zip` | `287058e58a8869998f51455d67f1c50ebd1d64f9226c7f564b4074825643c2a9` |

All 53 pages match clean source-archive builds at 85 DPI with the
separately supplied CVPR style. The changed pages were visually
checked. The archive reproduces the manuscript, not independent
confirmation. PDFs and ZIPs remain local.

[Experiment](../opendd_feasible_recipient_imitation_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md)
