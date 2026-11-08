# Evidence Manuscript v102

This draft adds the [gain-wait comparison](../opendd_gain_wait_v1/README.md)
and the eight-query menu audit. The learned timing rule loses to
immediate use at matched budgets. Most expiries are due to the support
clock, not an absence of sufficiently large directions. No independent
confirmation was opened and the original 2% rule is unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v102
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v102.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v102.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-gain-wait
```

Clean source-archive builds matched all 8 main and 28 supplementary
pages at 85 DPI with the same separately supplied CVPR style. Main
page 7 and supplementary pages 27 and 28 were visually checked.
The anonymous replay reproduced 171 count tables and 231 paired
intervals across 28 readouts. These checks establish reproducibility
of the scalar evidence, not independent performance.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `3c7fd2d329c375b698e853b70644912cb4c4f133a3490451bbfd1994de66c387` |
| `supplement/supplement.pdf` | `825ada7e9409cebc7e61d824fc984e58f86bed79dd59cc8e3c7d15d3716b3afc` |
| `manuscript_source.zip` | `6cf76ef760e8f714f7a299fef5f4c72d6986687752d41bc7bdbd6c8b6f9b2920` |
| `supplement_source.zip` | `a52bbc1559b42efbf5933d0ab4daead3a0aed36d8da62552838fd3bc7a3305d7` |
| `opendd_timing_replay_v15/anonymous_opendd_timing.zip` | `9ffc9ff1f06a59a2afb7d682f401b12ea9d28f4aa58a670828e393d968ec1c4b` |
