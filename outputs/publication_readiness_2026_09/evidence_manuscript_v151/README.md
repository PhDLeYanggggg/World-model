# Evidence manuscript v151

The main draft now distinguishes a lack of joint opportunity from a
failure to tune the joint selector. The supplement shows the controlled
replacement of proxy slots with actual streaming slots and the
double-slot-only capacity bound. Neither establishes an improved
deployment policy or independent confirmation.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v151
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v151.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v151.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `763ed0340314a4583340b71faf0550ed72d7b5e1575363db3b1363fa7bab62ab` |
| `supplement/supplement.pdf` (45 pages) | `f5a2d1a6d3646e2f4cc091b3d12fc9fbe9c1256bc537df7cb76ae4a3cb62474d` |
| `main_source.zip` | `e145dc114a8d5c9d0d59e83eaf7ab77ef85e5e7ad4d19e058209eaf1a0a50e9e` |
| `supplement_source.zip` | `e5d87f0f988fbabf45b860ebdcd94f18eeda08344921057756a9d403c5022da8` |

All 53 pages match clean builds from the two archives at 85 DPI with
the separately supplied CVPR style. Changed pages were visually checked.
The archives reproduce the draft, not training or scientific confirmation.
The PDFs and ZIPs remain local.

On 9 November 2026 the official [dates](https://cvpr.thecvf.com/Conferences/2027/Dates)
still list 10 November (registration), 16 November (main), and 23
November (supplement), all AoE. The [author guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
retain the eight-page body limit, anonymization, completed profiles
by registration and mandatory LLM usage disclosure. No registration or
submission action was taken in this experiment.

[Experiment](../opendd_stream_spline_calibration_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md)
