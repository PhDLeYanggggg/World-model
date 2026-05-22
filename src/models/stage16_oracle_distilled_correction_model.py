"""Stage 16 conservative correction model shell.

The executable implementation is intentionally lightweight and deterministic in
`src.stage16_pipeline` to avoid the Apple Silicon OpenMP/SHM failure path.
"""

MODEL_FORM = "prediction = baseline + intervention * bounded_residual"
