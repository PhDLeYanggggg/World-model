# Frozen Forecast Inference On Apple Silicon

I measured the same three checkpoint seeds and selector coefficients
as the registered calibration runner. The probe uses 28 existing
development queries (one median-sized and one largest per source),
2,357 eligible rows and 21 to 193 agents per query. The prefix geometry
is rebuilt and checked exactly against the frozen cache. No target
labels or external windows are read. The live calibration continued
alongside this short probe, so the timings include shared-host load.

| Torch threads | Batch | Three-seed core time (s) | Bitwise candidate parity | Changed actions |
| ---: | ---: | ---: | :---: | ---: |
| 4 | 256, start | 38.083 | yes | 0 |
| 2 | 256 | 33.958 | yes | 0 |
| 8 | 256 | 39.750 | no | 0 |
| 4 | 512 | 36.973 | yes | 0 |
| 4 | 1024 | 36.396 | yes | 0 |
| 4 | 256, repeat | 38.291 | yes | 0 |

At the registered setting, EqMotion takes about 96% of the measured
core time and all matched and screened decisions together take under
0.4%. The input packing takes 0.044 s in this already mapped development
probe; that excludes dataset I/O, hashes and setup. The two-thread
variant saves about 11% relative to the mean of the two baseline
passes. It is a plausible local optimization, not full-roster parity.
The eight-thread variant changes 699 EqMotion values, with a maximum
absolute difference of 0.000183 image pixels. Decisions and selected
paths happen to match on this sample. That is not sufficient to treat
the forecast bank as identical.

**Decision:** leave the active run and the registered confirmation
implementation at four threads and batch 256. No early scoring,
fallback backend, restart or new CREATE job was used. A much larger
speedup would require addressing EqMotion, not the joint search.
Confirmation is still conditional on complete calibration; the
profiling result is not scientific or deployment evidence.

```sh
.venv-pytorch/bin/python scripts/profile_m3w_frozen_inference.py --profile \
  --output data/stage_cvpr2027_experiments/frozen_inference_profile_v1/repeat.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_frozen_inference_profile.py \
  tests/test_m3w_dronecrowd_calibration.py tests/test_m3w_dronecrowd_confirmation.py
```

[Timing and parity receipt](profile.json). Nine targeted tests passed.
The real probe ran in native arm64 Torch 2.12.0 with interop threads 1
and zero loader workers, took 223.5 s for the six measured layouts,
and peaked at approximately 1.82 GB RSS. Timing varies by host load.
