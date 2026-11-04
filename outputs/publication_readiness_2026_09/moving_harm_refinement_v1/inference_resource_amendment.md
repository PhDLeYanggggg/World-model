# Phase-specific inference storage reserve

This is an operational correction, not a changed scientific gate. The inherited
9GiB reserve is larger than needed for this inference-only phase. All54refiners
are already frozen, and the observed complete training/private state plus
partial inference ledgers occupy less than10MiB. No optimizer, new feature or
latent cache, video, image, dataset download or full episode materialization is
created by evaluation or replay. The100MiB output cap remains in force.

The separate inference-only wrapper prospectively reserves5GiB on disk, while
preserving the100MiB artifact cap,12h boundary, CPU4/interop1/workers0, every
candidate, model, hash, random seed,0.25s/query solver budget,2%risk rule, source
bootstrap and complete evaluation roster. It cannot launch training. The
training reserve remains9GiB; the earlier interruption and exact resumed
training are retained unchanged. A reserve is an operational buffer, not a
model-validity criterion; conflating them would unnecessarily stop a small
metadata write while several GiB remain available. If5GiB is crossed, preserve
state and report the resource failure rather than lowering this limit again.

The already-running9GiB evaluation is not interrupted. Use this wrapper only
if that attempt encounters its resource guard, or for the subsequent exact
replay. Partial held results do not select this setting: it is justified solely
by measured artifact size and the absence of training/materialization. No
threshold or model is selected using partial results. Completed allocation
prefixes and group receipts must replay exactly; missing cells are completed,
not silently dropped or replaced. Record all interrupted attempts and actual
observed exit codes; resumed duration is not the entire experiment duration.

```bash
.venv-pytorch/bin/python scripts/resume_m3w_harm_inference_local.py evaluate
.venv-pytorch/bin/python scripts/resume_m3w_harm_inference_local.py verify
```

This does not convert an undefined risk into zero, open another data role,
recertify the exposed development results, execute Stage5C or enable SMC.
