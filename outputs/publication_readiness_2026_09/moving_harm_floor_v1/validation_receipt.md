# Validation Receipt, 5 November 2026

## Scope And Outcome

This receipt separates real training, engineering replay, synthetic regression
tests and scientific evaluation. The nine-fit harm-allocation experiment is
complete; the primary matched utility comparison and original safety gate fail.
The research goal is not complete. No deployment or submission claim follows.

## Actual Execution

All commands use native arm64 `.venv-pytorch/bin/python` from the repository
root, CPU 4/inter-op 1/workers 0. No new CREATE job or NumPy training fallback.

| Phase | PID | Wall seconds | Outcome |
|---|---:|---:|---|
| Three real interrupted-versus-continuous pilots | 9647 | 96.8153 | exit0; all exact |
| Initial training | 9913 | not recorded for whole failed process | exit1 at receipt export, after intact first fit |
| Exact recovery and remaining full fits | 10991 | 1252.7397 | exit0; all nine fits complete |
| Own-B weights and actual sampler counts | 13746 | 30.2944 | exit0; all nine exact |
| Fixed exposed-development evaluation | 13840 | 74.1903 | exit0; no reserved/H data |
| Initial/final metrics, attribution and held replay | 14065 | 78.5895 | exit0; exact |
| Posthoc own-B tradeoff diagnosis | 14367 | 32.7677 | complete; six decompositions checked |
| Tradeoff diagnosis replay | 14462 | 34.4788 | complete; all six exact |

Training includes 18,000 primary updates, 2,000 full-refit updates and 600 pilot
updates including repeats. The resumed training duration excludes the failed
process; the completed first fit took 182.7510 seconds. Peak resumed training
RSS was 7,588,331,520 bytes. The first checkpoint is byte-identical after recovery:
`e4bad40a0b2f320f87d97c43fc00697765d56fb99e638d8f48a64dae471858fd`.
The original registration and separate serialization amendment remain intact.

Primary analysis SHA256:
`5c1385b8190eea71abc00ecfd4c4bd9953a6512b4ff539c5495e0ec3bc299631`.

Posthoc tradeoff analysis SHA256:
`85509e65076b42bd5001c259cea59ce7538080d24a71fa657376fded1c6f6727`.

## Reproduction Commands

The original registered sequence and resume instructions are in
[operation_zh.md](operation_zh.md). The final report requires completed fits,
controls, weights and exact metrics replay; it fails closed on an incomplete
roster. The additional diagnostic has a separately bound posthoc registration.

```sh
.venv-pytorch/bin/python scripts/verify_m3w_harm_floor_weights.py
.venv-pytorch/bin/python scripts/run_m3w_moving_harm_floor.py verify
.venv-pytorch/bin/python scripts/diagnose_m3w_harm_floor_tradeoff.py --verify
.venv-pytorch/bin/python scripts/report_m3w_moving_harm_floor.py
```

These commands depend on local permitted historical data and checkpoints;
the public repository alone is not a complete data/weights distribution.

## Focused Regression Tests

Final tested version: **51 passed in 5.07 seconds**, exit0.

```sh
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_decision_semantics.py tests/test_m3w_zero_channel.py \
  tests/test_m3w_joint_decision_risk.py tests/test_m3w_joint_decision_safety.py \
  tests/test_m3w_joint_decision_reporting.py tests/test_m3w_composed_risk.py \
  tests/test_m3w_fixed_risk_readout.py tests/test_m3w_decoupled_risk.py \
  tests/test_m3w_decoupled_risk_safety.py tests/test_m3w_signed_decision.py \
  tests/test_m3w_full_objective_risk.py tests/test_m3w_harm_admission.py \
  tests/test_m3w_harm_floor_risk.py tests/test_m3w_harm_floor_reporting.py
```

Tests include causal-input isolation, unchanged unknown supervision, matched
positive coefficient mass, exact resume, legacy controls, failed-screen
readout blocking, checkpoint completion boundaries, JSON-only recovery,
registration binding and deterministic report figures. Synthetic fixture
tests do not establish real-data scientific performance; the real executions
above supply that separate evidence. The full historical suite was not rerun:
legacy tests can rewrite archived reports, and unrelated stages were not edited.

## Artifact Checks

- Regenerating the final report exited0 and left all 42 then-existing public
  experiment files byte-identical, including both SVG figures and metrics.
- Both PNG previews were visually inspected after moving the legend out of
  the plotting area. Axes, labels, curves and bars are readable without overlap.
  PNG previews remain private; public SVGs contain scientific plots, not data images.
- Root research_state numeric results match saved analysis and verification
  receipts exactly, including the negative contrast and failed safety counts.
- Scoped whitespace checks pass. Public outputs are summary tables, hashes,
  loss traces and documentation; private rows, arrays and weights are excluded.

## Scientific Boundaries

The original 2% risk/easy/support rules are unchanged. No identity-fallback
certification, posthoc threshold search or reserved20 Moving/DUT/DroneCrowd/H
readout. The original24-group primary stays NA; empty controls are not passes.
Missing-label completion bounds are not sampling CIs. Same-data exact replay
is not independent confirmation. Native obs8/pred12 image coordinates are not
metric, seconds-level, true3D, foundation or human gold. Stage5C and SMC remain
disabled. The paper addendum records a negative component experiment, not a
compiled submission update or a successful world-model contribution.
