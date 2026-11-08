# Evidence Manuscript v91

I added the [causal one-edit reserve](../opendd_reserve_v1/README.md).
The manuscript reports the increase in joint passes and the separate
loss of utility. It keeps the same-query comparison distinct from full
policies and the reserve distinct from the original budget.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v91
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v91.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v91.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-reserve
```

Local and clean builds with the same separately supplied CVPR style
match across all 8 main and 25 supplementary pages at 85 DPI.
Main page 7 and supplement pages 24 and 25 were visually checked.
The anonymous timing archive reproduces 52 count tables and 55
intervals in seven studies from anonymous scalars, not from raw
training. Generated PDFs, archives and weights remain local.
No independent confirmation or submission was made.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `37d3c08c7974107963ffa74575cdbd1570a7885a845d2ebbb16bc81986c985d7` |
| `supplement/supplement.pdf` | `1e341a68305026c5c23a645eb37004a51cba8bcc2a604bdb04809872d68aac68` |
| `manuscript_source.zip` | `72fc48eaa5ccf5eb79f547dbb30e4bdc288ea01a98555c8b4435d43a9c8585cf` |
| `supplement_source.zip` | `eed2adec88c1adb50976d238c08323c4e0f92512cabaf0b8d79df469770cb8e5` |
| `opendd_timing_replay_v4/anonymous_opendd_timing.zip` | `195c936e524dcbf48f8b46fa2e8b199f72a672bc9680cf062afc9a12bf02edd2` |
