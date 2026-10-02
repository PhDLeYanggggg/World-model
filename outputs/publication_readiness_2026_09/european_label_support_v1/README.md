# Frozen-action Label-support Diagnostic

The completed experiment links frozen cost-head errors to raw tracker rows and
observation quality. It finds that incomplete future labels do not explain most
selected harm. It does not prove tracking noise or improve a model.

- [Findings and numerical tables](conclusions.md)
- [Failure taxonomy and next repair boundary](failure_analysis.md)
- [Frozen protocol](protocol.md)
- [Machine-readable summary](summary.json)
- [Independent verification](verification.json)

## Reproduction

From the repository root, with the existing private source assets and checkpoints:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 .venv-pytorch/bin/python scripts/replay_m3w_label_support.py
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/verify_m3w_label_support.py
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m pytest tests/test_m3w_label_support_diagnostic.py tests/test_m3w_label_support_readout.py tests/test_m3w_observation_quality.py -q -p no:cacheprovider
```

The original order was register, pilot, run, verify, scalar verification. The
registration was committed as `1d5f0829` before real-data association. Completed
scientific JSONs are immutable. The replay wrapper executes the unchanged frozen
code and compares every raw join, prediction and aggregate with those JSONs.
Only the runtime receipt is redirected to a new private timestamped log. Its
schema and all non-runtime fields must match the original receipt; PID, elapsed
time, peak memory and free space may differ. Existing evidence is never replaced.

## Execution Evidence

| Run | PID | Terminal exit | Measured seconds | Peak RSS |
|---|---:|---:|---:|---:|
| Pilot: all raw rows, first head |30681|0|98.39|6.476 GB|
| Full 72-head diagnostic |30919|0|138.41|9.279 GB|
| Complete raw/inference/readout replay |31149|0|138.50|9.270 GB|
| Repeat-safe reproduction wrapper |31749|0|138.13|9.196 GB|

The wrapper's private runtime receipt is
`data/stage_cvpr2027_experiments/european_label_support_v1/additional_replays/1790952728817629000.json`,
SHA256 `d51de5ca0dacf2fbc8b8827baef328477822f424ca719788c72ecfcaecc25de6`.
It completed another full scientific equality check without replacing the first
replay receipt or any result. The original pipeline remains hash-bound.

Nineteen scoped tests pass. The full legacy suite was not rerun; unchanged
historical scientific pipelines are not newly certified. All 72 prediction,
action, row, target and envelope bindings match (504 head hash checks per run).
The separate scalar implementation checks 1,786 fields and four pre-existing
occurrence anchors. Raw-label masks and coordinates match all 318,969 admitted
rows on both full passes, including 3,221,201 valid future coordinate labels per
pass. The receipt's `valid_label_boxes` field counts requested centers recovered
from raw boxes; it is not an independent check of four cached future box corners.

Native arm64 CPU, four compute threads, single process, no DataLoader workers,
no MPS probing. No new model, checkpoint, numeric cache or HPC job. Local free
space was about 7.10 GiB, below the unchanged 10 GiB numeric-cache reserve, so
processing stayed in memory and wrote only about 1 MB of aggregate evidence.
The 9.45 GB source archive was streamed and read-only. Existing files, including
other projects' staged work, were not deleted.

## Evidence Status

- Models, splits and existing past-quality arrays: cached_verified.
- Raw future-label linkage, frozen inference and associations: fresh_run.
- New past-quality auxiliary training: not_run; next hypothesis, not a result.
- Visual identity/label adjudication: not_run; no independently adjudicated subset.
- Independent selection/calibration/confirmation: closed.
- Deployment upgrade, Stage5C execution and SMC: false.

The 3,000 bootstrap draws resample eligible localities, not windows or heads.
Eleven localities define the selected-harm share; nine define unmatched quality
contrasts, eight define paired contrasts. These are nominal exposed-development
intervals without multiple-comparison correction, not formal confirmation.
Task and labels remain obs8/pred12 raw stride12, image-local detector-silver.
No metric, seconds, human-gold, physical-safety, true3D, foundation or
submission-ready claim is made.
