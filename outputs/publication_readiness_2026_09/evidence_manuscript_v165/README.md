# Working manuscript v165

Figure 1 now gives each outcome its own labeled axis. The two panels use
the same scale, but separate hindsight capacity from past-only policy
results. Counts are printed next to the points. The supplement shows
seed recovery and matched-count gain as separate panels. Its interval
plot distinguishes known steps from the full-grid lower bound on the
same scale, with zero marked. Both figures are grayscale vector PDFs.

The experimental inputs, counts, intervals, tables, and manuscript text
outside the two captions are unchanged from [v164](../evidence_manuscript_v164/README.md).

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v165
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v165.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v165.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `e5cddab200a1f9218951fe217f5cc2464e95ef972309159b7a0a544a173a7330` |
| `supplement/supplement.pdf` (50 pages) | `fb80b7d4bf72bd2eb170248a5bb8d18424d8e9b3bc73f1d4e8a78a959fe76f63` |
| `main_source.zip` | `050f1cbc1442f465355f1f08a138487ed2beaa358b277e1d5afa91f34eb7fb5e` |
| `supplement_source.zip` | `0025fd70bd6d0670230bf005a6981ad8a7f202c49f300f2b0e8f889dab411de3` |

Both figure pages were inspected in the compiled PDFs. All 58 pages
matched the clean source-archive renders at 85 DPI with the CVPR style
supplied separately. The focused figure and manuscript tests passed
(25 tests). The PDFs and source archives stay local. Nothing was submitted.
