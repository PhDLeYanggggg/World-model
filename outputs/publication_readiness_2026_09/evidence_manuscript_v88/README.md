# Evidence Manuscript v88

I added the [recipient and timing experiment](../opendd_slate_queue_v1/README.md)
to the main manuscript and supplement. The result distinguishes
capacity from performance: the causal menu contains many useful
assignments at the same clock, but the fitted policy does not
identify them reliably. This is exposed development evidence.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v88
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v88.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v88.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py
```

Both sources were built locally and from clean extractions with
the same separately supplied CVPR style. All 8 main and 24
supplementary pages matched when rendered at 85 DPI. Main page 7
and supplementary pages 23 and 24 were visually checked.

The separate anonymous timing archive was extracted and replayed:
27 count tables and 21 intervals matched, with no recording names,
personal paths or confirmation data. It reproduces aggregation
from published scalars, not raw inference or training. Generated
PDFs, archives and weights remain local. No submission was made.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `4e2f69eb4ea4507ad2f761f1e2aeb87642ae9f20a992a2ffb18748371b705ce3` |
| `supplement/supplement.pdf` | `650df64c6fde430537895c1dbea732f468e43445d94392c3a659e1041ad115a7` |
| `manuscript_source.zip` | `1d86bdc73aaa70180659d0fac84b3a169c6cc841101e5d02d324c211b0f95e08` |
| `supplement_source.zip` | `f275064d32c11eb310940fb48a4ff4b0618dccef307788a7e1c3027353bb4686` |
| `opendd_timing_replay_v1/anonymous_opendd_timing.zip` | `d5f368206843f79fc6103fc1ee6265f67a18f7e3b59e6d3dbb22e33b45987dcd` |
