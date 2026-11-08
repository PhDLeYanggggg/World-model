# Evidence Manuscript v93

I added the [independent gain and support slots](../opendd_two_reserves_v1/README.md).
The main paper reports the gain and support trade-off at matched
queries. The supplement preserves the amplitude and reranking
controls and separates causal results from the exact timing bound.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v93
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v93.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v93.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-two-reserves
```

Local and clean archive builds matched on all 8 main and 26
supplementary pages at 85 DPI, using the same separately supplied
CVPR style. Main page 7 and supplementary pages 25 and 26 were
visually checked. The anonymous archive reproduces 86 count tables
and 100 intervals across 13 readouts from scalars. It does not
reproduce training. PDFs and archives remain local. No submission,
promotion, or independent confirmation was made.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `b10723a53bac7a41b0f8df0234f4e8f8b99f723d2650ca535aead35bc0aafeb6` |
| `supplement/supplement.pdf` | `2c3049cc6ab5eb051ce47c6814b1a3dc93dea1f686654b69461bde18e83f7158` |
| `manuscript_source.zip` | `c17a273d28bded17eecf83d680e1a9dd5afe9a1c8e8e37b3ec1764f503afcdab` |
| `supplement_source.zip` | `7ac4783f901eeaad141af3b752491d8577e2986e7449475752e8c8484688443c` |
| `opendd_timing_replay_v6/anonymous_opendd_timing.zip` | `be96c4ea102f3aea4a6d69a709c6a3412e24a2bd3830808aaba48d03f293b993` |
