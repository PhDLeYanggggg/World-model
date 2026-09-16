# Neural Benefit/Harm Regression on Matched OOF Inputs

## Material Passport

2026-09-16. Implementation and synthetic execution check. `fresh_run`: CPU/MPS cost-head fitting, recovery and synthetic development evaluation; source caches are hash-checked. No real forecasting fit, development result, risk calibration or confirmation. The real protocol remains unapproved and unchanged. This work enables the neural-versus-simple cost-head comparison; it does not claim that neural capacity helps.

## Gap Addressed

The backend previously defined `GainHarmHead`, but only its ridge alternative had a training and evaluation path. An untrained neural class cannot answer whether neural risk features are needed. `src/world_model/m3w_neural_gain_harm.py` now fits that head, and `scripts/train_m3w_neural_cost_head.py` reuses the ridge entry point's complete, verified OOF fold caches instead of constructing a different training dataset.

Every fold must appear once, each producer must exclude the whole held-out fold, and the feature/row/producer fingerprint is recorded. Targets have a separate digest: changing supervision invalidates resume even though the matched-input fingerprint deliberately excludes targets. Mean and scale are fitted only on fit-role OOF features. Inputs remain observed history, neighbors and frozen rollout diagnostics, not future positions or availability.

The trajectory forecaster is frozen. This work trains a downstream cost head, not the forecaster end-to-end, a residual head, JEPA or generative rollout. Existing Stage37 weights and historical accuracy tables are untouched.

## Objective

For the protocol's declared past-normalized error functional L:

```text
g = L(baseline, target) - L(candidate, target)
b = max(g, 0); h = max(-g, 0)
loss = mean over rows and two outputs of (predicted_cost - target_cost)^2
predicted_gain = predicted_benefit - predicted_harm
```

The GELU MLP has nonnegative Softplus outputs. It preserves cost magnitudes, without winner classes, clipped probabilities or per-row normalized weights. Internal coherence does not make predicted harm a calibrated probability, confidence bound or safety certificate. Squared regression targets conditional means in the population idealization, not a guarantee of accurate estimates under limited data or domain shift.

The protocol must explicitly provide `gain_harm_training` with `width`, `loss="squared_benefit_harm"`, and `fit_settings`: steps, batch size, learning rate, checkpoint and heartbeat frequencies. Its listed seed must agree with OOF producers. No values were added to the real draft. Fitting is fixed-budget, not validation-selected best; the separate development evaluator compares completed prespecified heads. There is currently one neural specification per protocol, not automatic hyperparameter search.

Report/checkpoint, normalization, producer architecture, seed, baseline and code identities are checked before development labels are opened. The five existing controls accept neural scores through the same interface as ridge. Calibration/final loaders pass the requested device and bind the new dependency; existing ridge calibration/final behavior is regression-tested. These compatibility checks are not a completed neural three-seed calibration/confirmation study.

## Recovery and Runtime

`latest.pt` stores weights, optimizer, sampler order/cursor/RNG, Torch RNG, losses, identities and runtime segments. Writes are atomic. The CLI publishes a risk-head artifact only after the fixed budget completes. Changed labels, features, producer/code identity or training settings refuse resume. Completed runs return `cached_verified` without further updates. No DataLoader multiprocessing, resource inventory probing or automatic device fallback is used.

CPU production CLI tests compared 12 uninterrupted updates against 5 plus 7 resumed updates with exactly equal parameters and losses. An explicit MPS test also had zero maximum parameter/loss difference and ran past-scene inference with the forecaster and cost head on MPS. This is short synthetic recovery evidence, not 12-hour stability or equality across CPU and MPS devices.

The first MPS attempt in the restricted environment failed at device initialization with a macOS-version error, while `sw_vers` reported 15.3.1. The same test passed under authorized Metal-capable execution, with no code change or CPU fallback. Both outcomes are retained. Production CLI CPU verification used arm64 Torch 2.12.0, compute threads 2, interop 1 and workers 0. Direct in-process tests inherit interop settings; they are not described as production-runtime measurements.

## Synthetic Result and Negative Finding

The CLI fixture used **64 OOF rows, 306 features and 12 neural updates**. Minibatch loss went from 0.5114436 to 0.3016263. These are different minibatches, not a validation curve or convergence proof. Neural and ridge heads share the OOF feature fingerprint and identical candidate forecasts at development evaluation.

The fixture has 27 past-supported queries but only 15 complete ADE/FDE labels in one synthetic scene. Ridge does not intervene. The neural guarded controls intervene on 5/27 queries (18.52%), but **all five lack complete future labels**. Their complete-label switch rate is 0%; scored improvement is 0%. Neither head passes the positive-gain rule, so selection keeps the floor. The uncontrolled predictor worsens mean error by 104.54% in this small fixture. No extra tuning was used to manufacture a positive example.

Decisions must retain agents without full futures, and coverage must be reported both before and after label filtering. This does not establish safe behavior on the unlabelled interventions. The existing risk screen treats missing-label interventions conservatively; it was not applied as a new neural calibration experiment here. One scene supplies no scene-bootstrap CI, and none was fabricated. There is no hard-slice support in the fixture.

## Checks and Reproduction

- Final neural/development/deferral/calibration/confirmation run: 95 passed and one opt-in MPS case skipped in 63.94 s. The earlier 94-pass run is retained in the ledger.
- 178 additional contract, admission, backend, solver, public-baseline and reader tests passed in 26.43 s.
- One explicit MPS test passed on the final code in 2.57 s; an earlier version passed in 3.63 s after the retained restricted-environment failure.
- The CLI integration test was rerun once (12.49 s) to retain light evidence after pytest rotated its temporary directory. This is not an independent experiment.
- Real neural-cost preflight returns exit 2 before training: `Explicit protocol approval required`.

A final regression exposed that an appended byte sequence in a completed checkpoint could be read and silently re-registered during resume. The new head now verifies its completed receipt hash before loading that checkpoint. The regression first failed, then passed in the final run. This repair is scoped to the new neural cost trainer, not a claim that every legacy checkpoint reader has been hardened.

```bash
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 .venv-pytorch/bin/python -m pytest tests/test_m3w_neural_gain_harm.py -q
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 PYTORCH_ENABLE_MPS_FALLBACK=0 M3W_TEST_MPS=1 .venv-pytorch/bin/python -m pytest tests/test_m3w_neural_gain_harm.py::test_mps_neural_cost_resume_and_shared_scene_inference -q
.venv-pytorch/bin/python scripts/train_m3w_neural_cost_head.py --preflight-only
.venv-pytorch/bin/python scripts/train_m3w_neural_cost_head.py --help
```

The first command skips the explicit MPS-only case. The second needs a Metal-capable execution context. Formal fitting needs an approved protocol, producer manifests, verified ridge OOF cache, listed seed and output directory; this document does not issue those decisions. See [machine evidence](synthetic_integration.json) and [verification](verification.json). No real data/cache/checkpoints are Git deliverables. The historical non-hermetic full suite remains 1,870 passes and one unrelated data-lake fixture failure; it was not rerun or relabelled green. CREATE access is unchanged and no job was submitted.

## Research Still Required

The decisive comparison requires the pending scientific protocol, clean source roles and matched real forecasters. It must test ridge, neural cost regression and literature-derived deferral on the same OOF observations, then compare intervention arms at matched coverage or explicit budgets. No neural superiority, calibrated risk, independent cross-domain success or CVPR contribution is established. Source eligibility and independent confirmation support remain unresolved. Stage5C and SMC remain off.
