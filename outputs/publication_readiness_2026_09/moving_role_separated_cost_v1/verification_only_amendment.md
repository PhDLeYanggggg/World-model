# Verification-Only Float/Jacobian Amendment

All 48 primary models have finished under training definitionv2. No policy
inference or outcome readout has run. Scientific model, inputs, losses, draws,
updates, thresholds and all 2%/support conditions remain unchanged.

A broader test run reveals that the reference-invariance test depended on the
unseeded synthetic weight draw: its strict bitwise comparison fails on2/6
values, maximum5.960464477539063e-8, relative1.1906010399798106e-7. The smaller
18-test run had passed. This is retained as a real failed check, not hidden.
Identical reference inputs can incur one-ulp float32 row/GEMM differences;
the test must not confuse bitwise row equality with mathematical action dependence.

Fix the synthetic seed and use8float32epsilon relative forward tolerance, with
zero absolute tolerance, for reference/probability only. Strengthen the check
with an exact zero Jacobian with respect to all70 candidate-dependent fields.
This checks the intended lack of action information, not only constant outputs.
No scientific risk allowance or actual prediction-replay tolerance is changed.
Actual cached predictions still require bitwise replay and all full refits
still require exact optimizer/model/trace agreement.

`registration_v3.json` governs verification tooling and retains the immutable
v2 training registration as the checkpoint identity. It explicitly verifies
that model implementation, data schema, protocol, config and settings equal
v2 before accepting any phase. No refit is relabeled a new primary model, and
no parameter, inference math, candidate, solver or readout is changed.
