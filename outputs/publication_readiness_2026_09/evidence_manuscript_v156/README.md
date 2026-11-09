# Evidence manuscript v156

The abstract now reports the upper bound for arbitrary edits
across an entire Moving recording at the old total displacement,
not the earlier single-query bound. The supplement derives the
bound, compares the two finite menus at both budget ceilings, and
reports what changes when missing-step debits are included.
The added passing view is a hindsight witness, not evidence of
online improvement or independent generalization.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v156
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v156.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v156.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `3e0d8b52c822e216006e2a49d771a7dfcd80ba7b2cf0dbe9c6b30f5f6600d339` |
| `supplement/supplement.pdf` (47 pages) | `f2579f014469255a0cda7d8cc301c1d17b658b086add89167c7b1b37c699b7f5` |
| `main_source.zip` | `aa2672c6fb0478e73d61ebdeda04c044f4933b7d2c4e4a2da0ccc21b08671386` |
| `supplement_source.zip` | `0d0082e3d627e3cbfe5ecbd923129f05868ff29e880fedc4fb93dc75e5dd9d36` |

All 55 pages matched the clean source-archive builds at 85 DPI
with the CVPR style supplied separately. Changed pages were
visually checked. PDFs, source ZIPs and per-action solver
checkpoints remain local.

The official [CVPR dates](https://cvpr.thecvf.com/Conferences/2027/Dates)
checked November 9 list enrollment November 10, full paper November
16, and supplement November 23, all AoE. The [author guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
require an LLM usage statement, present in the main paper and
supplement. This is a working draft; the capacity audit is not
independent confirmation and does not qualify the method for
submission on its own. No enrollment or submission was performed.

[Allocation audit](../moving_record_peak_allocation_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md)
