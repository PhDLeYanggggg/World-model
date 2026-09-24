# Bounded Independent Code Review

A separate reviewer inspected the numerical solver, failure-only repair runner,
raw-label verifier and focused tests. The reviewer did not change files, train
models, use external services or repeat the expensive full evaluation.

## Finding and Resolution

One P2 finding: the archival loader rebuilt source bindings without enforcing
equality to the frozen parent, and its verifier did not compare the reconstructed
repair identity with the saved repair identity. A memory-only counterexample
showed how source drift could be accepted. No actual drift was found in this run:
all 1,160 parent bindings and the inspected source hashes matched.

The supported entrypoint is now `scripts/run_m3w_easy_allocation_guarded.py`.
It checks every frozen parent binding, the parent identity hash, full repair
identity equality and actual source files before running. The new full verifier
uses the same guarded loader and writes a separate receipt, preserving the old
scientific source and audit record.

The reviewer rechecked this fix: seven new tests pass, five classes of entrypoint
identity/file drift are rejected, and the P2 is closed for the supported guarded
route. No new actionable issue was found. Original numerical review ran 26 tests
and found no additional scaling-equivalence, risk-check, failure-subset or
decision-label-input defect.

## Residual Limits

Direct invocation of the archival runner/verifier bypasses the added wrapper.
Those files remain historical evidence, not the supported operational entrypoint.
This is intentionally explicit rather than silently altering a completed
experiment's source identity. Public reproduction instructions use the guard.

Small-query enumeration in the full arithmetic verifier checks the independent
gain allocations, not every large unary/joint optimum. Synthetic solver tests
include all three objectives; actual large problems rely on checked solver
certificates. Code review and numerical checks do not establish risk calibration,
external generalization, a new best model or independent research confirmation.
