# Null Solver Certificate: Pre-Readout Engineering Amendment

The original decision process completed95 groups, then exited in group96 when
`float(result.mip_dual_bound)` received `None`. No new development outcomes were
read. All completed arrays and checkpoints are retained. This is not evidence
that a learned model succeeded or failed.

Only a missing dual certificate is converted from `None` to `NaN`. The already
registered solver checks then reject any uncertified solution and use their
existing feasible anchor fallback. Missing certificates are never zero and do
not authorize an unverified optimum. Objectives, budgets, node limit256, tie
handling, count matching, risk constraints and models remain unchanged.

Sealed source files are not edited. Function-local dependency namespaces reuse
the exact registered Python code objects while injecting this narrow adapter;
live module globals are not monkey-patched. The95 completed records are bound
before recovery and must stay byte-identical. Full108-group replay remains
mandatory, including the recovered group and every old control.

Synthetic regression tests first reproduce the original TypeError, then check
fail-closed behavior for unsuccessful and apparently successful results with
missing certificates, unchanged behavior with valid certificates, and untouched
legacy globals. Actual solver status/message and a causal-input hash are logged
for every missing-certificate event. This repairs failure handling, not the
underlying optimizer's numerical failure. Its numerical cause remains unproven.

Register and commit this amendment before recovery. Run the new recovery entry
for `decide` and `replay_decide`; use the original entry for `evaluate` and
`replay_evaluate`. The first decision runtime receipt measures the resumed
invocation, not the prior interrupted segment; the full replay supplies a
complete measured runtime. No retraining or outcome-based adjustment occurs.

Keep all development/source restrictions, undefined-risk views and original
gates. Independent confirmation remains closed; Stage5C/SMC remain disabled.
