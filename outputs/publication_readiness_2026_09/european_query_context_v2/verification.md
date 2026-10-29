# Verification and Reproduction

## Completed Evidence

- Registration commit2ea5884c preceded corrected fitting; training freeze
  commit18ab3abc was pushed before any corrected held readout.
- Corrected pilot PID59910 exit0:300 actual updates including exact50+50
  resume/direct100 replay,21.264s,2.579GB peak RSS. All encoder update and
  protected-base checks passed.
- Full training PID60147 exit0:36 fits,72,000 updates,381.689s,2.952GB RSS.
  All36 checkpoint/receipt hashes and trainable12,322/total25,471 counts verify.
- Frozen evaluation PID61062 exit0:72 views,95.521s. All eight old comparator
  predictions, policy scores and conditional MSE reproduce exactly.
- Full inference/readout/bootstrap replay PID61456 exit0:all72 views reproduce.
  It reads the same approved TRAIN packets, not independent data roles.
- Separate verification reconstructs all27 contrast/bootstrap summaries with
  scalar locality averaging and3,000 draws.57 aggregate checks and18 paired
  training checks pass. Initial states/objectives, RNG draws, budget, actual
  encoder changes and protected base parameters are checked.
- Benefit/harm/unknown decomposition reconstructs every full and matched
  contrast and exactly replays. Two scalar tests verify sign/denominator and
  equal-locality rather than repeated-row weighting.
-60 scoped tests pass in3.49s. No full legacy artifact-writing suite was run,
  because it can modify frozen historical research outputs.
- The plot was rendered from frozen aggregates and visually inspected. Image,
  data and weights remain local; only its checksum receipt is public.

The first `verify --check` reported a serialization mismatch: in-memory
parameter-count tuples versus JSON lists. JSON round-trip comparison isolated
this as the only difference. Returning explicit lists fixes replay without
changing any numerical artifact, checkpoint, bootstrap or decision.
This verification is not an independent research replication by another team.

## Existing-Artifact Checks

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_query_context_risk check
.venv-pytorch/bin/python -m scripts.verify_m3w_query_context --check
.venv-pytorch/bin/python -m scripts.diagnose_m3w_query_context --check
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_query_context_risk.py tests/test_m3w_query_context_readout.py tests/test_m3w_query_context_inference.py tests/test_m3w_query_context_diagnostics.py
```

These require the local hash-bound private models and approved CREATE TRAIN
packets. The public Git repository alone is not a standalone anonymous
reproduction bundle. Never delete completed receipts to claim another fresh run.
In a separately provisioned reproduction, follow register/commit/pilot/train/
freeze/eval/check; use `--resume` only for an interrupted incomplete phase.

## Runtime and Resource Boundary

Native arm64 PyTorch2.12.0, CPU4/inter-op1, DataLoader workers0. Atomic checkpoints
and heartbeat every100 updates;12h interruption/resume boundary,10GiB free reserve,
256MiB checkpoint cap. Private v2 directory occupies about19MiB after evaluation.
CREATE read-only inventory verified24 TRAIN packets; later file streaming worked.
The scheduler query timed out, so current queue state is unknown, not failed.
No CREATE submission, restart, authentication modification or cancellation occurred.

## Scientific Boundary

All12 localities are exposed original TRAIN. Source-preprocessing/whole-recording
separation is checked, but no independent confirmation claim is made. Unknown
labels are not imputed or used to filter inference. Frozen benefit/reference
outputs and test-use rules stay unchanged. Obs8/pred12 rawstride12, detector-silver,
local coordinates; no metric/seconds/physical-safety claim. No new forecaster,
deployment, Stage5C execution or SMC. Historical contaminated scores are not
recertified. The registered scientific advancement screen failed.
