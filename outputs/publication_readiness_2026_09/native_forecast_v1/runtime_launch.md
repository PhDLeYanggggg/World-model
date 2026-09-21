# Real Training Launch and Resource Decision

Status at launch, 2026-09-21; this is not a completed forecasting result.
The design/code was committed as c6c25a4d before real fitting. Native arm64
.venv-pytorch, Torch CPU four compute threads, one interop thread, workers zero.

The registered coupa/native/seed17 pilot completed 100 actual optimizer updates
in 1.9383 fit seconds (initialization and source verification additional).
Loss and gradients were finite; checkpoint persisted at step100. No held-source
score was read. The pilot is part of the 4,000-step endpoint, not a throwaway model.

The all-trial continuation started as PID59969 / execution session12888. First
old-loss endpoint completed at4,000 updates; the native endpoint then resumed
from100 rather than restarting. At a4m37s process check, CPU was192.2%, RSS about
1.1GiB, state running. There were3complete trial receipts. This snapshot does not
prove later completion; use live process/session and terminal receipts together.

Projected fixed-matrix runtime is roughly30-60minutes plus evaluation, based on
the actual pilot and completed trial rate. Local disk at launch had about62GiB
free. Local execution is reasonable for this small existing Transformer; no
GPU/multi-GPU infrastructure is needed for this discriminating loss comparison.
No CREATE job was submitted. Remote queue/assets were not freshly inspected;
the old access/path limitation is not evidence that the remote queue is empty.

36 scoped tests passed before launch, including exact synthetic uninterrupted
versus resumed parameter equality. This is supporting engineering evidence,
not forecasting improvement. The real run logs50-step heartbeats and200-step
atomic checkpoints under data/stage_cvpr2027_experiments/native_forecast_v1/.

All24registered endpoints must finish before any held-source scoring. No model,
seed, threshold or loss arm is chosen from partial held outcomes. No deployment,
independent confirmation, Stage5C or SMC. All windows remain offline supplied
annotation inputs, not strict real-time sensor-as-of reconstructions.
