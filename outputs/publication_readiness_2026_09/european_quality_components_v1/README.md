# Frozen Quality-Component Attribution

Complete diagnostic, no new model or deployment. See [findings](conclusions.md),
[failure taxonomy](failure_analysis.md), [protocol](protocol.md), [gates](gates.md)
and [next learning hypothesis](next_action.md).

## Reproduce

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/verify_m3w_quality_components.py
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/report_m3w_quality_component_cohorts.py
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m pytest tests/test_m3w_quality_components.py tests/test_m3w_quality_component_runner.py tests/test_m3w_quality_component_verifier.py tests/test_m3w_quality_component_cohorts.py -q -p no:cacheprovider
```

These reduce/check completed outputs; they do not rerun model inference. The
registered inference entry supports `register`, `pilot`, `run --resume` and
`verify`. Completed `run` refuses overwrite; `verify` repeats all frozen model
inference and arithmetic, checks immutable outputs, and writes a separate private
runtime receipt. Registration/source commit:`c65e635b`.

## Execution

- Read-only CREATE preflight verified owned storage with144 checkpoint files and
  no existing M3W scheduler jobs. Only72 quality files were needed for inference.
- Pilot PID36523 exited0 in23.0281s, peakRSS5,980,225,536bytes.
- Full PID36740 exited0 in134.0388s; all72 heads x13 variants completed,
  no scope reduction. PeakRSS9,376,595,968bytes.
- Native arm64 local CPU4/interop1,workers0. No new fitting, optimizer, tree
  splits, neural updates, scheduler jobs or remote scientific computation.
-72 immutable checkpoints,281,848,932bytes streamed into memory from
  `/users/k24101830/m3w/european_past_quality_auxiliary_v1/inputs/`.
  Every content hash and model identity matched the prior experiment.
- No local numerical row/weight cache. Group JSONs total6,972,402bytes, below
  the32MiB aggregate cap. Unrelated staged files and simulation work are untouched.
- Exact raw prediction and readout replay; original/quality prediction/action
  hashes and paired counts match.6,624 parent scalar checks.
-42,624 independent scalar/bootstrap fields verified.14 scoped tests pass;
  no claim to have rerun the full historical suite.

Summary SHA256:`b02df3db6b3e33a4ff010557bc9bdc8be94a114b213bf29f740ea3bfbe5c6137`.
Complete SHA256:`27e6cb74021179ff24e1b65da458cd0e01ef216d1fea5ef89d645e65e15fa710`.

All evidence remains exposed-source development, image-local detector-silver,
obs8/pred12 raw stride12. No metric/seconds, physical safety, human-gold, true3D,
foundation or submission-ready claim. Independent roles closed; Stage5C/SMC off.
