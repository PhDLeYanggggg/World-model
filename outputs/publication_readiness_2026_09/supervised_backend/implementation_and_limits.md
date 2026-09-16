# Causal Supervised Training and OOF Cost Bridge

Date: 2026-09-16. Status: implemented and engineering-verified; no new real forecasting result. The scientific protocol remains unapproved. This work does not substitute for the requested matched experiments or CVPR submission evidence.

## Gap Addressed

The legacy `src/m3w/train.py` reads old feature/teacher-based datasets. Its model names and completed historical runs do not establish clean supervision under the rebuilt recording protocol. The new recording reader, provenance contract and joint intervention solver previously had no connected supervised predictor/cost-learning backend.

The new path is:

```text
approved recording roles + verified raw-position cache
  -> fit-only causal forecaster with optimizer/checkpoint state
  -> frozen leave-fold-out predictions on other fit recordings
  -> realized baseline-relative benefit/harm labels
  -> fit-only normalized ridge cost-head control
  -> later development selection, independent calibration and confirmation
```

Only the first four computational connections are implemented here; the last scientific evaluation remains pending. The default real protocol is still `draft`. Both new training entry points refuse it before importing Torch or reading supervision. No fake approval was added to the real data configuration.

## Implemented Components

`src/world_model/m3w_supervised_intervention.py` provides a lazy fit/development dataset over the hash-checked reader, separate inference and label collation, a deterministic past-context Transformer, a coherent benefit/harm MLP interface, verified out-of-fold cost construction, and a simple ridge cost-head control.

The Transformer sees historical agent/neighbor positions, their actual past-relative timestamps, valid-history masks, causal baseline rollouts and requested prediction times. Attention can mix tokens inside the observed past; there are no ground-truth future tokens and no autoregressive latent rollout. Unknown inference fields and positive context timestamps are rejected. The output is normalized deterministic future displacement. A supplied baseline name is explicit; this backend does not label constant velocity or any other comparator the strongest baseline without the separate appropriate selection experiment.

The forecaster has a fixed training budget and masked coordinate MSE. It does **not** claim validation-selected best status. Its loss is an optimization objective, not official ADE/FDE. Variable requested horizons are padded only for batching; padded entries do not become target observations. Cost FDE is measured at the final *requested* step, not the last available label or the padded batch endpoint.

For baseline B, frozen candidate N, target Y and a past-derived positive scale s, the cost labels use the approved ADE or FDE of normalized trajectories:

```text
g = error(B, Y)/s - error(N, Y)/s
benefit = max(g, 0)
harm = max(-g, 0)
predicted_gain = predicted_benefit - predicted_harm
```

Benefit and harm predictions are nonnegative. Therefore predicted harm is at least the negative part of predicted gain. This is internal score coherence, **not** calibrated true risk or physical safety. Normalization is by observed path/motion scale, not by a future endpoint and not by test statistics. Raw dataset-local errors still need separate reporting in the eventual evaluation.

The ridge control fits its feature mean/scale only on OOF fit rows. It receives complete normalized past/neighbor context and summaries of baseline/candidate rollouts, not future labels, future-validity masks or oracle class IDs. The MLP cost head is implemented and its output coherence tested, but was not trained in a comparative real experiment.

## Provenance and Recovery

Before constructing cost labels, the contract verifies the producer and all declared parents against the entire held-out fit fold. The checkpoint's own recorded fit provenance must agree with its artifact manifest. The loader binds code identity and rejects an incomplete fixed-budget teacher. A predictor altered in memory after loading is rejected before cost construction. Duplicate OOF target identities are rejected.

Forecast checkpoints contain model weights, optimizer state, sampler order/cursor, sampler RNG, Torch RNG, configuration, code/protocol identity, runtime segments and losses. Writes are atomic. An interruption resumes from the last complete checkpoint, never a half-applied optimizer update. A completed identical run is returned as `cached_verified` without more gradient steps. Different training settings/baseline/code cannot be silently resumed as the same run.

OOF extraction records per-batch heartbeats and atomic per-fold caches. An interrupted fold is recomputed, while complete folds are hash-verified and reused. The cost-head CLI binds the fold/model map, producer hashes and ridge setting; it refuses altered fold caches and does not overwrite a completed identical head. These checkpoints are local generated artifacts, not Git deliverables. No multiprocessing, resource inventory probe, automatic CPU masquerading as MPS, Stage5C execution or SMC is used.

## Fresh Verification

| Evidence | Result | What it does not prove |
| --- | --- | --- |
| New backend tests | 17 tests | Not an experiment on real future targets |
| Combined focused regression | 102 passed in 11.65 s | Not the non-hermetic full suite |
| CPU within-process resume | Exact parameters, optimizer tensors, order, RNG and loss sequence | Not arbitrary-platform numerical identity |
| CPU separate-process resume | Exact final parameters and loss sequence | Not a long-run convergence test |
| MPS separate-process resume | 3 + 5 updates equals 8 uninterrupted updates; parameter/optimizer/loss max difference 0 | Not 12-hour stability, throughput ranking or real dynamics lift |
| Real raw-cache inference queries | 24 queries / 345 agents, finite and identical after future corruption | Not forecasting accuracy |
| Additional real query requests | 3 with no raw-t50 past-grid support, explicitly skipped | Not manufactured t50 data |
| Real draft training preflight | Both entry points exit 2 with explicit-approval refusal | Not a training run or new scientific permission |

The resume fixtures use varying turning trajectories and neighbor histories, not identical straight-line examples. They are explicitly synthetic, temporary fixtures with test-only approval markers. Those markers do not approve real studies. One test-helper path-type error occurred during development and was fixed without weakening its checkpoint-byte assertion; the final current tests pass.

The real input check uses random Torch weights solely to test the full inference interface. Future-label methods are replaced by rejecting stubs; zero such calls occurred. It corrupts every agent's positions after the query frame and verifies agent membership, packed inputs, forecasts and risk features are unchanged. These checks use the existing cached-verified raw-position sources. No original dataset was rewritten and no real target was used for training, model selection or error computation.

Evidence: [real-input checks](real_input_checks.json), [MPS recovery](mps_resume_checks.json), [verification ledger](verification.json). The unchanged historical full suite remains 1,870 pass / 1 unrelated data-lake fixture failure; it was not rerun or relabelled all-green here.

## Runnable Checks

```bash
.venv-pytorch/bin/python scripts/check_m3w_supervised_inputs.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_supervised_intervention.py -q
.venv-pytorch/bin/python scripts/train_m3w_causal_forecaster.py --preflight-only
.venv-pytorch/bin/python scripts/train_m3w_oof_cost_head.py --preflight-only
```

The final two commands intentionally return exit 2 for the current unapproved draft. Both CLI `--help` interfaces expose the actual fit arguments; the technical model defaults are in `configs/m3w_intervention_backend.json`. No approved protocol, fit group, baseline floor or seed is invented by that configuration. Fitting commands additionally require explicit recordings, baseline, seed, output path and an approved protocol. `--resume` preserves identity; `--device cpu|mps` chooses a backend explicitly. Cost-head extraction additionally requires the explicit held-fold-to-model mapping and producer manifests.

## Remaining Scientific Work

1. Approve the main horizon, metric/aggregation, data roles and applicable risk budgets. Existing exposed recordings remain exploratory development material, not untouched confirmation.
2. Fit real clean forecasters and training-only baseline floors within the approved folds, then generate real OOF cost supervision. Do not reuse old test-exposed teachers.
3. Implement and execute development-only selection of forecast checkpoints and cost-head/policy candidates. The current fixed-budget fit must not be described as validation-selected.
4. Compare a strong simple selector, neural cost head and matched public predictors; then compare uncontrolled, independent, scene-uniform and joint intervention with the same forecasts and matched budgets.
5. Execute independent calibration/confirmation with sufficient actual scene support, three formal seeds and cluster-aware uncertainty. The present model/provenance interfaces cannot prove IID or correct human declarations.

No new deployment promotion. No real-world gain, calibrated-risk guarantee, true 3D, foundation, metric/seconds-level, or CVPR-readiness claim. CREATE access and the scientific decisions remain unresolved; local engineering work did not require a remote job.
