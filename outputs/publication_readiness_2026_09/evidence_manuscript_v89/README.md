# Evidence Manuscript v89

I added the [same-clock recipient-ranking experiment](../opendd_slate_rank_v1/README.md).
It reports improvement over fixed ranking without implying superiority
to trained regression, constant velocity, or the original full-roster
risk requirement. The separate sensitivity check is labeled post hoc.
No independent confirmation or deployment is claimed.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v89
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v89.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v89.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-rank
```

Both sources build locally and from clean extracts with the same
separately supplied CVPR style. All 8 main and 24 supplementary
pages match at 85 DPI. Main page 7 and supplement page 24 were
visually checked. The separate anonymous archive verifies 33
count tables and 33 intervals across four timing/ranking studies,
without raw trajectories, site names or local account paths.
It reproduces aggregation, not training. PDFs and archives remain local.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `c7c061617fdb830ebecba0bac798c2363440ad357d50728f46e3fbfc98dc2fd3` |
| `supplement/supplement.pdf` | `a9c8e22e130e1f5b9d01f1c237eb797b2e8f86e9c038668ffc755597af691824` |
| `manuscript_source.zip` | `22c6e1936df3b12509f07d7718d57f6de5d53ac7f11ca2adda060d0779870950` |
| `supplement_source.zip` | `82c801bfb41f533bcfff93036966dea974ff66160a034953cd5a900c985a79dd` |
| `opendd_timing_replay_v2/anonymous_opendd_timing.zip` | `f4b15a477c60ef18897773f9fa64dd3d14ee7905add6b94bac2c2fc68a0cb61a` |
