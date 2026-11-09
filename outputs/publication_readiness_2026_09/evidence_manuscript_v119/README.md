# Evidence Manuscript v119

The main paper now reports the
[DUT stronger-baseline result](../dut_matched_motion_v1/README.md):
OLS4 outperforms frozen EqMotion on the full complete-label roster.
The supplement explains the nonoverlapping matched-motion comparison
and why its zero signed easy change is not supported risk. No new
independent evidence or submission claim is made.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v119
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v119.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v119.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

All eight main and 33 supplementary pages matched clean source-archive
builds at 85 DPI with the same separately supplied CVPR style. The
changed main pages and the supplementary table were visually checked.
The separate DUT anonymous package verifies 13 arm tables and
21 descriptive intervals, without predictions or trajectory data.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `2aec07e3cfbbb5316262568fedefa50f09c39cb30f13b5fb1616feae0249e2bf` |
| `supplement/supplement.pdf` | `03061a4fc6c7e90f790ae559ad05e3a49ee04614588855d841d94864bcb35eb2` |
| `manuscript_source.zip` | `25741af18237278ad707ee0d9dc15163fbbc99d930d8f8cea2212de389d24a37` |
| `supplement_source.zip` | `caaba74034517322d25cf7f92c322925d2faa4ec137eb7145d49c6a2d5cbf232` |
| `dut_matched_motion_v1/anonymous_dut_motion.zip` | `498802f904dc184dc46f8b6b518d560d1e54643226861998078ea73a9ccbeae1` |
