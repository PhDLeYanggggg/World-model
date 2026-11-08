# Evidence Manuscript v96

I added the [risk transitions](../opendd_support_monotonicity_v1/README.md)
to the main text and the
[whole-recording training comparison](../opendd_record_support_v1/README.md)
to the supplement. Better error in a single comparison is not a
risk pass or an independent transfer result.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v96
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v96.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v96.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-record-support
```

All 8 main pages and 27 supplementary pages from the clean source
archives matched the local render at 85 DPI with the same CVPR style
supplied separately. Main page 7 and supplementary pages 26 and 27
were visually inspected. The extracted anonymous archive independently
recomputed 116 count tables and 135 intervals across 17 readouts from
scalars. The recorded checkpoint replay also completed with zero
optimizer updates. Neither replay constitutes independent confirmation.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `b9c91443372120e80dce3392a6be62040181f0d94710d16768d1c4259842ca91` |
| `supplement/supplement.pdf` | `ccae43b1c57f4800c9ebc1a641a1cd468d0082b7824aa64ad873cfae56cd0c92` |
| `manuscript_source.zip` | `007220edc7fbf279d944ae29a642f92374b459ab081da26740bf6571fb1e4236` |
| `supplement_source.zip` | `0f3004a4bee0a1a1add5bb67beddb5576b8d22ea6642673dbf03f6b5807a25e2` |
| `opendd_timing_replay_v9/anonymous_opendd_timing.zip` | `d437d765da2ce2bc880a869085be84d5cd29d78907ce9cc180fa441eb76dd56c` |
