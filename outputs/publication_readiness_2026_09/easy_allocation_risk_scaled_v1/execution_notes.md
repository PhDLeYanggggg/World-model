# Execution and Verification Record

## Scope and Provenance

2026-09-24, native arm64 .venv-pytorch, local CPU4 / interop1 / workers0.
No new training, multiprocessing loader, GPU resource probe or remote job.
CREATE's last recorded access failure is not a current queue inspection.

The fixed first 256-query pilot took 0.610455 seconds in its allocation loop.
The completed V1 decision pass logged 293.339478 seconds; its complete replay
logged 288.539295 seconds. These exclude loading, hash checks and evaluation.
They are neither training time nor total end-to-end wall time.

V1 process PID78400 and replay PID79167 ended successfully. Evaluation PID79097
and aggregate replay PID79576 also ended successfully. All 188,388 query/action/
seed instances and all 175,756 target rows were processed; no scale downgrade.

## Failure and Repair

Original matched-control postchecks rejected 127 queries: 17 damping, 93
Transformer and 17 EqMotion. All invalid proposals failed closed. Two causal-only
diagnostic programs reproduced 14 fixed cases and inspected proposed primal
solutions. Every sampled rejection was an original-unit risk overrun; integer
and product errors were zero. The original result is retained, not rewritten.

The repair was registered before recomputing decisions. A positive cost-unit
factor multiplies gains, expected positive harms, geometry weight and budget.
Coherent risk max(harm,-gain) scales by that same factor, as does the complete
objective. Binary support, feasible decisions and minimizer ordering do not
change. Primal/dual values and risk are checked again in original units.

PID79855 repaired 127 queries in 5.649726 seconds of the repair loop; PID79908
replayed them in 4.706299 seconds. All 188,261 other query decisions remain exact.
There are 877 changed agent/arm bits, not 877 independent people. Evaluation
PID79952 and aggregate replay PID80025 completed. HiGHS printed internal diagnostic
lines at process shutdown but returned 0; no hang or process cancellation occurred.

The copied allocation and query functions are deliberately versioned to preserve
the original experiment's source hashes. AST comparison verifies identical bodies;
the new module substitutes only the numerical solver dependency. No scientific
risk tolerance or model parameter is changed. Private repaired caches reference
unchanged original chunks rather than copying the complete data population.

## Checks Completed

- V1 exact decision and aggregate replay.
- Repair exact decision and aggregate replay.
- Separate raw-label evaluation and original-unit predicted-budget recount.
- 188,388 query/action/seed instances, 14,236,236 decision bits checked per version.
- 1,190 fixed small-query optima checked by brute force; 17 larger checks skipped.
- 2,592 scene reductions and 45 paired contrasts recounted.
- All 127 repairs tied to original causal failure flags; all 188,261 successful
  original queries preserved. No metric-based repair subset.
- Initial numerical repair: 108 scoped tests, 1.95 seconds.
- Provenance-guard extension: 115 scoped tests, 1.96 seconds, including seven
  explicit parent/source/config/runtime identity-drift regression tests.
- All required processes terminal. The full unrelated legacy suite was not rerun.
- Figure preview inspected; tables and uncertainty remain readable.

The export whitespace check reports trailing blank lines in the hash-bound
versioned allocation module and the receipt-bound guard module. They are retained
rather than changing completed provenance for a cosmetic edit. README/state whitespace
checks pass. Strict JSON parsing and local report-link checks pass; the largest
public artifact is a 1.79 MB aggregate metrics JSON, not a per-row data cache.

The independent arithmetic script is independent code in the same research
workflow. It is not an external replication or independent confirmation study.

## Exact Reproduction

Independent code review identified a provenance-guard gap in the archived repair
loader. No current drift was found: all 1,160 parent bindings were preserved.
The supported operational entrypoint now requires every frozen parent binding,
the parent identity hash and full reconstructed repair identity to match their
saved records. Source/config/runtime drift is rejected. The archived runner and
verifier remain unchanged to preserve completed experiment hashes; do not use
those historical entrypoints as the guarded operational interface.
The [bounded review](code_review.md) records the original P2, its closure on the
supported route and remaining historical-entrypoint limitation. One concurrent
check attempt was correctly refused by the exclusive process lock while aggregate
replay ran; it did not interrupt evaluation and was retried after completion.
Guarded decision replay, aggregate replay, identity preflight and the complete
raw-label arithmetic verifier all subsequently passed. Their new receipts do
not replace the original verification records or change any reported metric.

Run from the repository root with the private, hash-bound local identities and
assets present. These are replay commands, not reconstruction from Git alone:

```bash
.venv-pytorch/bin/python scripts/run_m3w_easy_allocation_guarded.py --phase check
.venv-pytorch/bin/python scripts/run_m3w_easy_allocation_guarded.py --phase repair --verify
.venv-pytorch/bin/python scripts/run_m3w_easy_allocation_guarded.py --phase evaluate --verify
.venv-pytorch/bin/python scripts/verify_m3w_easy_allocation_guarded.py
.venv-pytorch/bin/python scripts/report_m3w_easy_allocation_repair.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_frozen_allocation_guard.py tests/test_m3w_scaled_risk_controls.py tests/test_m3w_easy_allocation.py tests/test_m3w_interaction_controls.py tests/test_m3w_joint_intervention.py tests/test_m3w_native_joint_controls.py -q
```

Existing decisions resume by hash. Verify replays require exact equality. Never
delete a binding or overwrite an immutable result to make a changed run pass.
No raw trajectories, private arrays, model checkpoints or third-party media are
included in the public commit. Public aggregate results alone do not reconstruct
the underlying private asset chain.
