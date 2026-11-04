# Aggregate Prior Serialization Fix

The first full training process PID31569 exited1 after its first2000-step
product model and its independent2000-step exact refit. The fit receipt tried
to serialize the7x4 fitting-only event-prior NumPy array with plain JSON.
The observed exception was `TypeError: Object of type ndarray is not JSON
serializable`. No nested-C validation had run. This was a result-writing
exception, not OpenMP/SHM, a training hang, or a measured model failure.

The original registered scientific files and registration stay unchanged.
`scripts/run_m3w_moving_joint_harm_runtime.py` converts only the top-level
`event_prior` aggregate array to its numerically identical nested list before
the existing serializer. It does not alter checkpoints, loss, initial weights,
samples, inputs, normalization, budgets, priors or evaluation rules. The
wrapper/test/this note are hash-bound in runtime_patch.json and committed before
resume. Other unsupported arrays still raise rather than silently changing
their representation. A focused regression test verifies the conversion and
the absence of input mutation.

Use the runtime entry for remaining phases:
`.venv-pytorch/bin/python scripts/run_m3w_moving_joint_harm_runtime.py train --resume`.
The completed first model/refit are checked and retained, not restarted. Use
the same entry with evaluate or verify after the complete training freeze.
Original pilot execution is still valid and is not repeated.
