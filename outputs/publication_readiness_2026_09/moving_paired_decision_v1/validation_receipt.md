# Validation Receipt, 5 November 2026

## Scope And Scientific Outcome

Nine paired-decision neural fits, frozen development readouts and exact replay
are complete. The registered rank-minus-point utility comparison is negative
and the original full-policy safety contract fails. The research goal remains
incomplete. No model promotion, independent confirmation or submission-ready
claim follows from these execution checks.

## Actual Execution

All runs use the existing native arm64 `.venv-pytorch/bin/python`, CPU4 threads,
inter-op1 and workers0. No NumPy substitute or new CREATE job was used.

| Phase | PID | Wall seconds | Outcome |
|---|---:|---:|---|
| Three real interrupted-versus-continuous pilots | 16418 | 110.7874 | exit0, all exact |
| Initial training | 16893 | not observed for whole process | terminated; last heartbeat update1500; exit code unavailable |
| Unchanged resume and complete training | 18194 | 1717.6010 | exit0; nine full fits |
| Fixed exposed-development readout | 20797 | 67.6353 | exit0 |
| Fitting, discrimination and held replay | 20940 | 75.3623 | exit0; all exact |
| Reusable completed-replay entry point | 21646 | 74.0829 | exit0; all36 public files unchanged |

The resumed duration excludes the initial process and downtime. No failure
cause is inferred from a lost terminal session. The implementation validates
the complete checkpoint binding before restoring model, optimizer, sampler,
RNG and update count. See [recovery](interruption_recovery.md).

Completed budgets: 18,000 primary updates, 2,000 full-refit updates and 600 pilot
updates including repeats. Peak resumed training RSS: 7,354,515,456 bytes.
All three joint controls exactly reproduce predecessor states and predictions.
The first full paired-rank refit is exact, as are all three recovery pilots.
All nine initial/final fitting metrics, error attributions and discrimination
statistics reproduce. Held predictions, actions and metrics also reproduce.

Analysis SHA256:
`f82f0e69fbd99dec60f1650dd102a234509d936828b963dfaa4ae7d77eb9380a`.

Full-refit checkpoint SHA256:
`10cff952a90fc630d57c7b6aed08dbb40de286b95e1889a0836234a93b7529f3`.

## Commands

The original fixed sequence and recovery commands are in
[operation_zh.md](operation_zh.md). Completed training phases are immutable.
Verification and report reconstruction use the existing private assets:

```sh
.venv-pytorch/bin/python scripts/replay_m3w_moving_paired_decision.py
.venv-pytorch/bin/python scripts/report_m3w_moving_paired_decision.py
```

The original runner refuses to overwrite a completed verification receipt.
The separate completed-replay entry point requires the complete original run,
refuses to create or replace public receipts, checks all saved metrics and
predictions, and preserves the entire public manifest. Only private runtime
events/heartbeat are appended. Do not delete original receipts to rerun a
phase. Report reconstruction is idempotent. Cloning the public code does not
provide private data/weights.

## Focused Regression Tests

Final combined model/report run: **66 passed in 5.41 seconds**, exit0.

```sh
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_decision_semantics.py tests/test_m3w_zero_channel.py \
  tests/test_m3w_joint_decision_risk.py tests/test_m3w_joint_decision_safety.py \
  tests/test_m3w_joint_decision_reporting.py tests/test_m3w_composed_risk.py \
  tests/test_m3w_fixed_risk_readout.py tests/test_m3w_decoupled_risk.py \
  tests/test_m3w_decoupled_risk_safety.py tests/test_m3w_signed_decision.py \
  tests/test_m3w_full_objective_risk.py tests/test_m3w_harm_admission.py \
  tests/test_m3w_harm_floor_risk.py tests/test_m3w_harm_floor_reporting.py \
  tests/test_m3w_paired_decision_risk.py tests/test_m3w_discrimination_audit.py \
  tests/test_m3w_paired_decision_reporting.py tests/test_m3w_paired_loss_geometry.py
```

The later completed-replay utility adds **2 passed in 1.28 seconds**, exit0:

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_paired_completed_replay.py
```

These two cases reject incomplete experiments and missing/changed receipts
without creating or overwriting them. Its real74.08-second replay additionally
preserves all36 then-existing public files; it neither retrains a model nor
produces a new scientific readout. The original66 tests remain valid for the
unchanged model version; these are68 distinct focused cases across two runs.

The scope includes same-query pairing, unknown/zero exclusion, causal inference
isolation, exact restart, unchanged controls, incomplete-roster rejection,
failed-screen readout blocking, deterministic scientific figures and analytical
loss geometry. Synthetic tests are not real-data performance evidence. The
full unrelated historical suite was not rerun. Subsequent report edits only
clarify prose; actual report regeneration and artifact checks follow below.

## Artifact Checks

- All seven registered scientific file bindings remain byte-identical.
- Repeating the final report leaves all 35 then-existing public experiment
  files byte-identical, including result_summary, metrics and the SVG.
- The actual PNG preview was visually inspected: axes, labels, legend and
  curves/bars are readable without overlap. The preview remains private;
  the public SVG is a scientific plot, not a dataset image.
- Root research_state numerical results agree exactly with the saved analysis
  and verification receipt, including the negative contrast and failed safety.
- Public content consists of source, configuration, reports, summary metrics
  and a scientific plot. Raw rows, histories, predictions and weights are private.
- The separate paper addendum does not change the compiled main manuscript.
  No new main-PDF compilation or visual review is claimed in this experiment.

## Scientific Boundaries

Original 2% all/easy positive-harm, signed easy-net, reference/support and missing-
label requirements remain unchanged. No identity certification, threshold
search or reserved20 Moving/DUT/DroneCrowd/H readout. Empty controls/support
are NA, not passed. Original 24-group primary remains NA. Completion bounds
are not sampling CIs; replay is not independent confirmation. Native obs8/pred12
image-coordinate results are not metric, seconds-level, true3D, foundation or
human gold. Stage5C and SMC remain disabled.
