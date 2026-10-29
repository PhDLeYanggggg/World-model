# Reproduction and Evidence Boundaries

The repaired protocol and implementation were frozen in0b8edf97 before v2
execution. Parent registration, training freeze, readout and all36 checkpoint
hashes are verified. v1 interruption and numerical reproduction are also bound.
Original trained models, causal inputs, source normalizers, sampler RNG states,
dual values, AdamW moments, threshold and2% risk definitions remain unchanged.

```bash
.venv-pytorch/bin/python -m scripts.audit_m3w_query_conditioning check
.venv-pytorch/bin/python -m scripts.verify_m3w_query_conditioning --check
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_query_conditioning.py tests/test_m3w_query_conditioning_verifier.py tests/test_m3w_query_context_risk.py tests/test_m3w_query_context_readout.py tests/test_m3w_harm_jacobian_audit.py tests/test_m3w_rational_harm.py
```

Completed artifacts are immutable. On a new authorized environment, use
`register`, commit the matching registration/code, then `run`; `run --resume`
is only for an interrupted incomplete run, not for deleting completed evidence.
Packets must be available through the established approved CREATE route and
the original frozen checkpoints must be present. This is not a standalone
public-data download recipe or external independent reproduction.

v2 PID65213 exit0;141.240318s;2.948GB RSS. Raw logs/heartbeat are private under
`data/stage_cvpr2027_experiments/european_query_conditioning_v2/`. No large new
cache or checkpoint. Thirty targeted tests pass; the full artifact-writing
legacy suite was deliberately not run. The original assertion failure was
reproduced before repair and its partial output is preserved.

Verification reconstructs all summary arithmetic from frozen receipts, checks
all fixed self/peer sample hashes, and compares12 completed v1 fit views and4
probe groups exactly. This is a separate implementation in the same repository,
not external scientific replication. The local four-panel figure is derived
from these aggregates and visually checked; no source image or raw trajectory
is committed. The earlier three-panel draft is retained as a presentation draft.

Authorizes no change in data roles, main horizon, scientific endpoint, risk
tolerance or formal submission. Routine audit/implementation is carried out
within the user's standing authorization; formal submission remains separate.
