# Paired Past-Quality Auxiliary Experiment

Complete; useful development prediction signal, failed safety advancement.
See [conclusions](conclusions.md), [failure analysis](failure_analysis.md),
[registered protocol](protocol.md), [gates](gates.md) and [next action](next_action.md).

## Reproduce

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/verify_m3w_past_quality_auxiliary.py
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/analyze_m3w_past_quality_auxiliary.py
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m pytest tests/test_m3w_past_quality_auxiliary.py tests/test_m3w_past_quality_readout.py tests/test_m3w_past_quality_diagnostic.py -q -p no:cacheprovider
```

These commands check/reduce the completed records; they do not train new models.
Registration was committed at `d0f4a432`, pilot/runtime receipt at `07e56288`.
The training entry supports `register`, `pilot`, `run --resume` and `verify`.
Completed `run` refuses overwrite. `verify` repeats the real fits and inference,
checks immutable scientific outputs and writes a separate private runtime receipt.
Incomplete resume refits/checks existing groups instead of silently replacing them.

Training was run with native arm64 `.venv-pytorch`, four compute threads,
interop1 and workers0. Both actual and placebo arms were fitted twice, compared
exactly, then serialized/reloaded for exact inference replay. No new Torch
gradient updates or tree splits: these are actual fitted conditional cost heads.
The same-model refit is not an independent implementation; scalar/CI reduction
has a separate implementation.

## Resources and Artifacts

- Full PID33623 exited0. Runtime690.8369s including remote verification;
  fit plus exact refit399.1598s, inference136.7358s, peakRSS9,428,140,032bytes.
- All72 source groups /144 auxiliary checkpoints completed, no scope reduction.
- Checkpoints563,739,641bytes, all remote hashes verified, below1GiB cap.
- Owned storage: `/users/k24101830/m3w/european_past_quality_auxiliary_v1/inputs/`.
  `checkpoint_manifest.json` binds every file. Original hash-bound forest and
  preprocessing are also required by `m3w_past_quality_auxiliary.predict`.
- Local group JSONs total967,039bytes. No new local numerical arrays or weights;
  available local disk7,634,358,272bytes remained below10GiB cache reserve.
- No new Slurm job or remote scientific computation. CREATE was used for owned
  file storage/checksums only; unrelated simulation jobs were untouched.
- 13,248 scalar/parent checks,440 independent aggregate/bootstrap fields, four
  original count anchors. 19 scoped tests pass; full historical suite not run.

Summary SHA256: `ee061bfc639cd0dd4cd004bd79484808a8e957b6729c8608972a70911188d121`.
Complete receipt SHA256: `6b52038aa96296c11c48184edad01e08aada90c19fa922f48f9757ed3d3de258`.

Scope: previously exposed development data,12 European localities,
obs8/pred12 raw stride12, image-local detector-silver. No independent confirmation,
metric/seconds, human-gold, physical-safety, true3D, foundation or submission-ready
claim. Stage5C and SMC remain off. These are not deployment checkpoints.
