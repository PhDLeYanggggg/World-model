# Evidence Manuscript v103

This draft adds the [fixed-gain early-support experiments](../opendd_early_reference_v1/README.md).
The local model replays the original. Recording-level advantage
supervision adds and advances support, but loses already defined
references and does not pass the original risk rule. The matched
comparison does not show an advantage over easy ranking.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v103
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v103.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v103.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-early-reference
```

Clean source-archive builds matched all 8 main and 29 supplementary
pages at 85 DPI with the same separately supplied CVPR style. Main
page 7 and supplementary pages 28 and 29 were visually checked.
The standalone anonymous replay verified 181 count tables and 253
paired intervals across 30 readouts. Independent confirmation
remains sealed; this is not a submission-readiness result.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `5ddf7965384358468ec8fa8f7d1915b815a2f9aa98dd1b7a023100137ce9dad1` |
| `supplement/supplement.pdf` | `b9d626dff42361343247c6b9341cf115d9e4cdcaf570a285a7e90af685eb8006` |
| `manuscript_source.zip` | `ec4583e91fe11cf3ff8d9cc9a7cf7a6d5b4421a121493d7e7d5ee4d498d9ec3a` |
| `supplement_source.zip` | `432330d5edc805bdb415501e3781aebc96770376555cbc7c92d182076ea53ba1` |
| `opendd_timing_replay_v16/anonymous_opendd_timing.zip` | `7ba0c72307695938af1e721ec96154f39a8afbdcdfc1b2155fa4702a4cd1037f` |
