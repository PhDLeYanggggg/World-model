# Prospective Atomic Storage Repair

Before the full policy grid or any calibration labels were constructed, a
checkpoint review found a two-file crash window in the first allocation
wrapper. Replacing choices.npy before its progress JSON could leave the old
progress hash pointing to new choices after an interruption. The failed resume
condition is reproduced in a focused test; no actual scientific result was
lost or changed by this issue.

The new wrapper commits the completed grid prefix, choices and binding metadata
in one fsynced atomic compressed checkpoint. Per-query checkpoints are still
handled by the unchanged original allocator. A separate immutable storage
registration binds the new code before execution. The original input/model
registration and all completed input/inference artifacts remain unchanged.

There is no model, normalizer, multiplier, objective, cohort, budget or solver
change. The original complete largest-record pilot can be hash-verified and
reused. A new real largest-record interrupted/resumed versus direct full-grid
pilot must match both one another and that original pilot before full allocation.
Its status is recorded in allocation_storage_pilot_v2.json when actually run;
this prospective repair note is not evidence that the pilot has already passed.

For complete allocation, use:

```bash
.venv-pytorch/bin/python scripts/run_m3w_calibration_atomic_allocation.py run
```

Then the original labels/eval/verify operators consume the same policy schema.
Calibration labels and final confirmation remain closed at repair registration.
