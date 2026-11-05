# Verification Receipt, 5 November 2026

## Real Experiment

- All six registered models completed 2,000 updates, with both neural encoders
  updated. All three fresh controls reproduce their predecessors exactly.
- Both 50+50 versus 100-update pilots and the first full 2,000-update treatment
  repeat match model, optimizer, sampling state and predictions exactly.
- Evaluation PID6819 exited 0 after 63.683569 s. Verification PID7216 exited 0
  after 69.122364 s. Six initial/final fitting metrics and joint attributions,
  held predictions/actions and metrics replay exactly.
- Frozen analysis SHA256:
  `f9b631a1c52686b7ea792c31e3ef3e7671d17b8b0a11b9bae9451607099f8296`.

## Focused Tests

The following check completed with **38 passed in 3.82 s**, exit 0:

```sh
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_decision_semantics.py \
  tests/test_m3w_zero_channel.py \
  tests/test_m3w_joint_decision_risk.py \
  tests/test_m3w_joint_decision_safety.py \
  tests/test_m3w_joint_decision_reporting.py \
  tests/test_m3w_composed_risk.py \
  tests/test_m3w_fixed_risk_readout.py \
  tests/test_m3w_decoupled_risk.py \
  tests/test_m3w_decoupled_risk_safety.py \
  tests/test_m3w_signed_decision.py \
  tests/test_m3w_full_objective_risk.py
```

Coverage includes zero-label gradients, positive-target dominance, unchanged
unknown supervision, causal-input isolation, exact resume, complete fit roster,
and frozen-action-before-outcome readout. These tests and real-data replay are
separate evidence. The full historical repository suite was not rerun; this
receipt is not a claim that every unrelated test or pipeline passes.

## Report Consistency

The report-generation command and its separate regeneration both exited 0.
All 27 experiment files existing at the regeneration check were byte-identical
afterward, including training freezes, metrics and the generated summary.
Nine numerical/status/hash fields in the current research-state entry matched
their saved evidence exactly; held policies and contrasts in the summary
matched analysis.json. Original-rule and no-promotion flags were checked.
This receipt was added afterward and is not among those 27 files.

The paper addendum is not included in the compiled main PDF. Public artifacts
are code/configuration, aggregate metrics and prose only; private arrays,
weights and third-party data remain local. Local permitted inputs are still
required for replay. No anonymous raw-data cold start is claimed.

## Scientific Verdict

Full original held risk/support fails, 0/17. Worst defined easy-net degradation
is 8.633364%, above 2%. Completion bounds are not sampling CIs. The original
24-group primary remains NA; independent confirmation is not_run. No identity
certification, deployment change or submission-readiness claim. This is
obs8/pred12 image-coordinate intervention-risk training, not new JEPA or
trajectory dynamics. Stage5C and SMC remain disabled.
