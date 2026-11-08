# Evidence Manuscript v100

I added the [fixed-budget ranking experiment](../opendd_budget_rank_v1/README.md)
and the frozen swap attribution. The text distinguishes the
relative-path ablation from the failure to beat the original
policy and records why easy-reference coverage changed.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v100
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v100.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v100.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-budget-rank
```

Clean archive builds matched all 8 main and 28 supplementary
pages at 85 DPI using the same separately supplied CVPR style.
Main page 7 and supplementary pages 27 and 28 were visually
checked. The anonymous package recomputed 151 count tables and
192 intervals across 24 readouts. Checkpoint replay separately
used zero optimizer updates. These checks do not establish
independent performance or replace upstream fitting.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `f7ca2eb7608b0d33dc7914c0370a126f6429ad2a4e3d2c328a4fb11d18f3dfd0` |
| `supplement/supplement.pdf` | `740f4b2d023a11314a92418674e3b3d1bcf216eee515ecca9ee56241cc6d9456` |
| `manuscript_source.zip` | `a8140ee01b3afd3435e1f2cd18e312a07ef265c25c3c87927cab674a6542d5cc` |
| `supplement_source.zip` | `4ff27e5e344251f609512e81c3f05c86d48c67db31dbcaf19281656592d1d2ba` |
| `opendd_timing_replay_v13/anonymous_opendd_timing.zip` | `17f49f4e34d02d018340c34c00f5e6e6dcabf7f600c9f98cf43228a88acbe2c2` |
