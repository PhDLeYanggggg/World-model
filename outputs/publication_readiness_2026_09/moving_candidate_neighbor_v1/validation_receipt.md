# Verification Receipt

## Actual Runs

Native arm64 `.venv-pytorch`, CPU4, interop1, DataLoader workers0.
No resource probing or new CREATE job. No current run left live.

| Phase | PID | Observed exit | Seconds | Peak RSS bytes |
|---|---:|---:|---:|---:|
| Initial registration failure before fitting | 28662 | 1 | not recorded | not recorded |
| Real probe and frozen-neural diagnostic | 28898 | 0 | 40.423874 | 6457311232 |
| Complete coefficient/prediction/metric verification | 29002 | 0 | 39.514030 | 6487572480 |
| Completed read-only replay | 29269 | 0 | 39.273771 | 6420905984 |

The completed read-only replay left all 16 then-existing public files byte-
identical. Documentation added later is not included in that file count.
All three context metrics, sixteen saved probe fits and frozen neural
attributions reproduce exactly. The parent neural checkpoints are unchanged.

Analysis SHA256:
`ee5b862b6095621e1cb83511c39796e0a180a3139f9130647ef92238e215fef6`.

## Focused Tests

15 tests passed in 2.25s across the candidate-neighbor geometry/probe,
reporting and existing rollout-history neural tests. They cover geometry,
neighbor masks/order, coordinate changes, training-only preprocessing,
held-label independence, original cost semantics, exact frozen attribution,
resume and input-label separation. The full historical suite was not rerun.
The diagnostic figure was rendered and inspected.

## Evidence Limits

Deterministic replay verifies implementation, not independent scientific
generalization. The fitting screen fails in all three contexts. There is no
new neural model, new V/H readout, support exception or deployment result.
Existing original-rule failure remains a failure. Stage5C and SMC remain off.
