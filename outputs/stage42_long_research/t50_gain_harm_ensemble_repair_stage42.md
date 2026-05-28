# Stage42-II T50 Gain/Harm Ensemble Repair

- source: `fresh_stage42_ii_t50_gain_harm_ensemble_repair`
- generated_at_utc: `2026-05-27T23:57:27.305922+00:00`
- input_hash: `58ca8e94b967e251db26ebd5bdf22d728784c6adce28bd969855f2c6090f7223`
- gate: `15 / 15`
- verdict: `stage42_ii_ensemble_repair_stabilizes_t50`

## Purpose

Stage42-IH showed that simply adding same-family selector seeds did not make ADE t+50 seed-stable. Stage42-II tests a validation-only score ensemble over six t+50 gain/harm selectors plus a three-checkpoint Stage42-N dynamics ensemble.

## Claim Boundary

- dataset-local/raw-frame 2.5D only
- no metric or seconds-level claim
- no true 3D/foundation claim
- no Stage5C execution
- no SMC

## Fresh Ensemble Test Metrics

| metric | value |
| --- | ---: |
| ADE all | 0.121192 |
| ADE t50 | 0.081363 |
| ADE t50 row CI low | 0.074234 |
| ADE t50 row CI high | 0.088162 |
| ADE t100 raw diagnostic | 0.141041 |
| ADE hard/failure | 0.124775 |
| ADE easy degradation | 0.000000 |
| FDE t50 | 0.209983 |
| FDE t50 CI low | 0.202250 |
| switch rate | 0.219637 |
| TrajNet t50 | 0.164168 |
| ETH_UCY t50 | 0.062896 |
| UCY t50 | 0.000000 |

## Per-Domain ADE

| domain | rows | all | t50 | hard/failure | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 25901 | 0.101778 | 0.062896 | 0.106084 | 0.005424 | 0.252152 |
| `TrajNet` | 20087 | 0.232755 | 0.164168 | 0.238808 | 0.000000 | 0.282023 |
| `UCY` | 9540 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Interpretation

- This is a fresh replay/evaluation, not new training.
- Base predictions and selector scores are cached as Stage42-II local intermediates after first computation; final policy selection and test evaluation are still freshly recomputed from those intermediates.
- The policy is selected on validation only, then evaluated once on test.
- If this passes, ensemble selection is a stronger deployable t+50 repair than any single seed.
- If this fails, the blocker is a model-family or domain-specific TrajNet t+50 issue, not seed count alone.
