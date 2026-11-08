# Evidence Manuscript v98

I added the [matched direction comparison](../opendd_dual_direction_v1/README.md)
and the sensitivity to omitting the largest contributing site.
The supplement reports both the positive CV-only comparison and the
failure to beat frozen neural directions. No independent evidence
or change in deployment was added.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v98
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v98.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v98.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-dual-direction
```

All 8 main and 27 supplementary pages matched the clean archive
build at 85 DPI using the same separately supplied CVPR style.
Main pages 7 and 8 and supplement page 27 were visually inspected.
The anonymous scalar package recomputed 134 count tables and
164 intervals across 20 readouts. The checkpoint replay used zero
optimizer updates. Neither check constitutes independent confirmation
or reproduces the upstream training from the anonymous bundle.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `8a033c5cebd28e8457c0a587436f98db17e7d0fc6542b747145b7226aece0d18` |
| `supplement/supplement.pdf` | `1706157439a3672141cad9f7fa67aac6b56311d178ac5e52bb616d2867fd575c` |
| `manuscript_source.zip` | `748c2a96435ec95697df0c339074e4cea0dbe8b414ec810f0d01cf04512ceafa` |
| `supplement_source.zip` | `0ae822500cd6c31082282d230d990bde5254aea5e599a184ef40fed0ce2f791b` |
| `opendd_timing_replay_v11/anonymous_opendd_timing.zip` | `4db7e1ddfa19d8c489395574c8d7cb1580d6667d0fc60c159e15891bac3cb9f4` |
