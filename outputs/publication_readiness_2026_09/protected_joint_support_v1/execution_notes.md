# Protected Joint Support: Reproduction

2026-09-22. Native arm64 `.venv-pytorch`, single-process audit. No model training,
GPU/HPC submission or numerical-runtime fallback. All 1,349 source bindings
verified in preflight (PID 41710, exit 0). Registration pushed as `188170f7`.

Audit PID 41762 exited 0; replay PID 41859 exited 0. The full pass wrote 99
view/recording receipts, with an identity-bound checkpoint after each recording.
Interrupted runs reuse the same receipts; `--verify` recomputes every query and
requires exact immutable output equality. A live sample showed 99% CPU and about
1.46 GiB RSS, not a peak-memory or speed benchmark. No task process is left live.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_protected_joint_support.py --audit-only
.venv-pytorch/bin/python scripts/audit_m3w_protected_joint_support.py
.venv-pytorch/bin/python scripts/audit_m3w_protected_joint_support.py --verify
.venv-pytorch/bin/python -m pytest tests/test_m3w_protected_joint_support.py tests/test_m3w_native_joint_controls.py
```

37 tests passed. The new 13 tests cover a genuine product-induced change, weak
coupling, a unique harm-feasible assignment, count/unsupported structural nulls,
explicit enumeration-cap refusal, identity/future-input rejection, allowed-only
array access and six independent scalar exhaustive checks. The other 24 tests
exercise the inherited joint-control implementation. No full legacy-suite claim.

Real-query enumeration checks all 266 non-additive opportunities, 61,024 subsets
and 18,521 feasible subsets with direct versus decomposed objective equality.
Replay checks 62,796 scene/seed queries and 188,388 pool/query records. This is
not a second scientific team or independent confirmation dataset.

Private per-query receipts remain in
`data/stage_cvpr2027_experiments/protected_joint_support_v1/`; only code, config,
reports, aggregate JSON and the replay receipt are committed. No target labels
or row cache is uploaded. Model/policy/thresholds and the pinned manuscript
snapshot are unchanged. The existing unrelated staged changes remain untouched.

CREATE project/authentication and remote assets are still unverified. This small
CPU audit did not require remote compute; no inference about absent HPC data or
jobs follows. Submission preparation remains incomplete.
