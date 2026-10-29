# TRAIN Aggregation Diagnostic: Complete

Fresh_run: 144 frozen-model TRAIN-monitor recomputations in 50.10s.
Model checkpoints and source packets are cached_verified. All input hashes,
causal eligibility, fixed monitor draws and saved soft-excess values replay.
No optimizer step, development-label access or independent-role access occurred.

| Arm | Decision | Risk | Query excess /72 | Agent-mass excess /72 | Query pass, mass fail /72 | Reverse /72 | Undefined reference /72 |
|---|---|---|---:|---:|---:|---:|---:|
| unit_control | soft | all | 38 | 48 | 15 | 5 | 0 |
| unit_control | soft | easy | 61 | 64 | 3 | 0 | 0 |
| unit_control | hard | all | 25 | 32 | 10 | 3 | 0 |
| unit_control | hard | easy | 52 | 54 | 3 | 1 | 2 |
| budget_scaled | soft | all | 13 | 12 | 2 | 3 | 0 |
| budget_scaled | soft | easy | 12 | 14 | 2 | 0 | 0 |
| budget_scaled | hard | all | 6 | 6 | 1 | 1 | 8 |
| budget_scaled | hard | easy | 3 | 5 | 2 | 0 | 9 |

Each view uses the same fixed 128 known-TRAIN query draws as training monitoring.
These are repeated query occurrences, not independent observations or a full
TRAIN census. The agent-mass calculation pools selected harm and reference on
these draws; it is not a new development evaluation or missing-label completion.
The original 2% budget is unchanged. Zero reference is undefined, not safe.

Query pass/mass fail directly exhibits different feasibility judgments under the
same predictions and action weights. The reverse can also occur. It does not
show how much development failure the mismatch causes. Soft/hard mismatch,
unknown outcomes, model error and domain shift remain separate limitations.
No arm is selected or promoted by this diagnostic. Positive scaling preserves
the query constraint zero set, not equivalence with pooled risk. Any new training
repair must be separately registered and retain the scientific readout definition.

Recheck aggregates without inference or training:

```bash
.venv-pytorch/bin/python -m scripts.report_m3w_risk_aggregation --check
```

The project remains dataset-local, raw-frame and 2.5D; no true3D, foundation,
metric/seconds or independent generalization claim follows. Stage5C and SMC stay
disabled. A risk-proxy diagnosis is not new trajectory dynamics.
