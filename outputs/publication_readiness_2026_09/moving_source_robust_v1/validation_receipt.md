# Execution And Verification Receipt

Observed on 5 November 2026; runtime verification and scientific success are
reported separately.

| Phase | PID | Observed exit | Seconds | Evidence |
|---|---:|---:|---:|---|
| Real pilot | 30383 | 0 | 73.5124 | Both 50+50/100 resume comparisons exact |
| Training | 30621 | 0 | 590.1975 | Six 1,000-update fits plus 1,000-update exact refit |
| Evaluation orchestration | 31462 | 0 | 16.7797 | Failed screen recorded; no new held predictions |
| Metric verification | 31505 | 0 | 39.8311 | Six initial/final fitting and source metrics, attributions exact |
| Completed read-only replay | 31790 | 0 | 41.1988 | All 26 then-existing public files unchanged |

The analysis SHA256 is
`a2f97022f6768523489a4b85f857d204acdb45970d8c98d4ead032827c83f8e2`.
Training peak RSS is 7,739,359,232 bytes. All registered code/config/protocol
bindings remain unchanged. Native arm64 Torch, CPU4/inter-op1/workers0;
atomic 100-step checkpoints and heartbeat. No duplicate or CREATE job.

## Focused Tests

Three non-overlapping batches give 45 passing tests:

1. Source-robust risk/reporting and parent rollout-history risk: 16 passed in 3.04s.
2. Original selection-forecast, reference-support and policy-conditional readout: 13 passed in 1.63s.
3. Fixed readout, decision semantics, rollout-history readout, joint and paired risk: 16 passed in 2.93s.

Tests cover exact unchanged control loss/gradient, complete source queries,
original unknown bounds, causal-only inference, resume and partial-update
failure, source-metric decomposition, no held readout after a failed screen,
NA reporting and original support rules. They do not constitute real-data
generalization evidence. The entire historical suite was not rerun.

The generated figure was visually inspected as a rendered PNG. Curves and
cost bars are legible without overlap. The lightweight SVG is public; rendered
PNG, input arrays and checkpoints stay private. The complete read-only replay
creates only a private execution receipt and heartbeat, not new public results.

The existing compiled manuscript PDF remains unchanged, SHA256
`d9babe3ddbac858bcc1ac30e3ba98b332bed625c78ff9231d1d42a3e30b11772`.
The new addendum is separate from that PDF.

## Scientific Status

Internal fitting screen 0/3; additional nested-C/H evaluation not_run. Prior
17-view risk/support failure and original 24-group primary NA remain unchanged.
No identity certificate, changed 2% limit, removed support condition or
deployment upgrade. Stage5C/SMC remain disabled; submission_ready remains false.
