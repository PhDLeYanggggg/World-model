# Evidence Manuscript v97

I added the [support-first scheduling comparison](../opendd_support_schedule_v1/README.md)
and the exact two-role capacity analysis. The main paper distinguishes
the increase in hindsight capacity from the lack of online benefit.
The supplement records the strong-baseline and equal-radius shape
diagnostics. No independent evaluation is claimed.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v97
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v97.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v97.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-support-schedule
```

Clean-archive and local renders matched on all 8 main and 27
supplementary pages at 85 DPI with the same separately supplied
CVPR style. Main page 7 and supplementary pages 26 and 27 were
visually inspected. The anonymous scalar replay verified 126
count tables and 145 intervals across 19 readouts. The forecasts
and training are not reproduced by this scalar bundle. The PDFs
and source archives are kept locally; nothing was submitted.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `a3bfdda82dca772e47891db771fb8ec5a03318d0a6da7ceb8b0d4807418163ec` |
| `supplement/supplement.pdf` | `cfb73d57aa447e01be8fc3a53ea25b12d651d791a9cc2038ae05cb8ff2e7234b` |
| `manuscript_source.zip` | `8d04a424e3d2fdac711a68264966e2831f11ea9d507fc58f716edf355303a114` |
| `supplement_source.zip` | `c977f95fc0ce73dc7674eb3df890a712e99511cd11d1dcd6925fc2e4fc357c7e` |
| `opendd_timing_replay_v10/anonymous_opendd_timing.zip` | `469d4680cac0fc8ece1e3b2b8c60e88249a62f55633287376d4e0cad25526155` |
