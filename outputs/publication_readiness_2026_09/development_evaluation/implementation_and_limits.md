# Development Selection and Matched Intervention Evaluation

Date: 2026-09-16. Result source: `fresh_run` implementation/synthetic checks; `cached_verified` raw-position inputs for engineering inference. Real forecasting evaluation, calibration and confirmation: `not_run`, scientific protocol unapproved. No new deployment or CVPR-readiness claim.

## Progress and Purpose

The preceding continuation made substantive progress: a resumable causal forecaster and verified out-of-fold benefit/harm learning path. This continuation connects those fitted artifacts to actual development-only model comparison. It is no longer necessary to use the historical test-exposed teacher/cache path to implement selection. The real scientific choices are still not filled in to make an experiment start.

New entry points:

- `src/evaluation/m3w_development_evaluation.py`: inference, label-only scoring, grouped summaries and guarded development selection.
- `scripts/evaluate_m3w_development.py`: frozen candidate family, checkpoint/report verification, recording-level recovery and selection provenance.
- `scripts/check_m3w_development_inputs.py`: real-input counterfactual checks with random weights and synthetic scores, without real labels.
- `tests/test_m3w_development_evaluation.py`: 17 focused regression and end-to-end checks.

## Fair Comparison Boundary

Every candidate forecaster and risk head is frozen before opening development labels. Producer checkpoints and all declared upstream artifacts must pass the existing physical-scene/role checks. The risk head's report is content-bound and checked against its checkpoint, OOF normalization source, fit recordings, parent models, metric and baseline identity. A test-statistics normalization declaration is rejected before label access.

For each candidate, the same observed agents and forecast arrays feed five controls:

| Control | Decision |
| --- | --- |
| Floor | Keep the explicit causal baseline for every agent. |
| Uncontrolled | Use every finite neural forecast, without gain/harm threshold gating. Invalid forecasts still fall back. |
| Independent | Optimize unary predicted gain under the common predicted-harm/intervention caps, without pair costs. This is budget-constrained unary selection, not fully uncoupled decisions. |
| Scene-uniform | Switch the entire compatible scene query only when supported, feasible and preferable. |
| Joint | Optimize the same unary scores/caps plus the declared excess-proximity pair cost. |

The baseline is supplied explicitly. No claim that constant velocity or another supplied baseline is the strongest is inferred from this interface. Strong-floor selection and published predictor comparisons are still separate experiments.

Identical caps do **not** imply identical actual intervention rates or realized risk. Reports preserve actual rates, budget violations and proximity proxies. Floor and uncontrolled are anchors, not budget-matched guarded policies. A matched-coverage frontier and independent risk calibration have not been executed. The predicted budget is not a safety guarantee.

## Past-Only Queries and Label Support

Scene query enumeration uses observed recording timestamps and continuous past support, not the future-complete target index. An agent with no future labels still participates in inference and intervention coverage. Predictions and decisions are computed before the label API is called.

Different native forecast grids are kept separate. Even at the same requested raw horizon, agents sampled every five and ten frames cannot silently share a forecast-time array. The evaluator records compatible query groups separately; it does not interpolate or claim cross-grid joint coordination. Past histories with insufficient support remain excluded by the causal reader, not by future completeness.

The caller must explicitly approve either complete-requested-path or available-step ADE. Missing final requested timestamps never become last-available-point FDE. Both ADE/FDE eligibility counts and the denominator for all-past-supported intervention rates are reported. Labels cannot reorder agents or change the requested endpoint.

## Metrics and Selection

The implemented pooled selector uses **explicitly approved past-normalized** ADE or FDE. Raw dataset-local ADE/FDE are additionally reported separately for each recording. Raw values from unverified, incompatible coordinate systems are not pooled into a single number. A different primary error-unit protocol requires an explicit extension, not an automatic switch to normalized results.

Aggregation follows the approved contract: equal physical scene, equal recording, or agent-window weighting. Equal-scene aggregation pools eligible agent queries within each physical scene and weights scenes equally. Equal-recording aggregation weights recording means equally. Bootstrap always resamples complete physical-scene blocks, carrying all their recordings/windows together and preserving the chosen weighting. It reports at least 2,000 resamples when at least two eligible scenes exist; a single scene returns `not_run_insufficient_physical_scenes`, not a fabricated narrow interval. Independence and adequate statistical power are not proven by the IDs or by having two scenes.

The development table reports baseline/selected errors, mean signed excess, mean positive harm, easy/hard slices defined relative to the baseline error, label coverage, intervention rates, per-scene and per-raw-horizon breakdowns, descriptive tail error, worst-scene mean error and paired scene-block intervals. Proximity is an uncalibrated dataset-local proxy, not a collision or physical-validity certificate.

The candidate with lowest approved primary development error is selected only if it improves the floor, meets the easy-degradation rule and respects predicted budgets. No easy support means no promotion. Otherwise the selected policy remains the floor. Ties use lower intervention and stable identity. The easy definition must match the protocol's risk definition, not an independently redefined convenient slice. Selection itself confers neither calibration nor deployment approval. Development intervals are descriptive and cannot correct selection optimism or become independent confirmation.

## Resume and Provenance

Before the first development label access, the CLI freezes the complete candidate family, all artifacts, approved protocol, implementation hashes and runtime settings in `run_identity.json`. Each finished candidate/recording result gets an atomic row cache and a hash receipt. A resumed run reuses only intact complete recording results; an interrupted recording is recomputed. Heartbeats record PID, candidate, recording, frame, row progress and elapsed time.

The completion receipt binds the report, selected policy and artifact manifest. Repeating an identical completed run returns `cached_verified`; changing a model, threshold family, implementation or cache bytes is rejected. The selected artifact explicitly declares development selection exposure and its producer parents. It cannot be represented later as a model never exposed to development data. The module never opens calibration or confirmation roles.

## Verification

| Evidence | Fresh result | Scope |
| --- | --- | --- |
| New evaluator tests | 17 passed | Unit, negative-boundary and synthetic end-to-end checks |
| Combined focused regression | 119 passed in 16.80 s | Not the non-hermetic full suite |
| End-to-end temporary fixture | Three 8-update forecasters, two held-fold cost sets, ridge cost head and five development controls | Synthetic engineering only |
| Synthetic development fixture | 16 compatible scene queries / 27 agent queries, only 15 with complete ADE/FDE labels | Missing future agents retained for decisions |
| Synthetic selection | Guarded controls returned floor; uncontrolled was worse | No positive empirical method claim; no real training |
| Single-scene synthetic CI | Explicitly not_run | No pseudo-independent bootstrap |
| Real-input counterfactual check | 23 queries / 274 agents, all five decisions/forecasts invariant | Random forecast weights and constant synthetic gain/harm only |
| Real unsupported requests | 4, explicitly skipped | No fabricated raw50 samples |
| Real future-label calls | 0 | No real accuracy or risk calibration |
| Current draft CLI | Exit 2, explicit approval required | Refusal before evaluation, not runtime failure |

The synthetic CLI test verified identical completed-result reuse, then intentionally corrupted its temporary row cache and verified refusal. Its temporary directory is therefore an adversarial fixture, not a reusable clean experiment asset. Re-running the test reconstructs it. The full suite's earlier 1,870 pass / 1 unrelated data-lake failure is unchanged; no full-suite success is claimed here.

See [verification ledger](verification.json) and [real-input evidence](real_input_checks.json). Existing CPU/MPS optimizer-resume evidence for the unchanged forecaster remains in the preceding supervised-backend report; it was not relabelled as a new neural accuracy experiment.

## Reproduce and Continue

```bash
.venv-pytorch/bin/python -m pytest tests/test_m3w_development_evaluation.py -q
.venv-pytorch/bin/python scripts/check_m3w_development_inputs.py
.venv-pytorch/bin/python scripts/evaluate_m3w_development.py --preflight-only
.venv-pytorch/bin/python scripts/evaluate_m3w_development.py --help
```

The preflight currently returns exit 2. Real fitting/evaluation requires user-approved scientific rules and roles first. The protocol needs an explicit `development_evaluation` section: selection error unit, label coverage policy, query stride, per-recording proxy geometry, easy/hard definitions, selectable control arms, bootstrap seed, solver limit and declared policy grid. The evaluated candidate-family file then references fitted artifact identities and content-bound cost reports. No real values were assigned in this continuation.

Next: obtain the pending scientific decisions, refit clean forecasters and OOF risk supervision, and run the actual development comparison. Then independently calibrate the frozen policy family and confirm it with sufficient physical-scene support, three formal seeds and matched strong controls. Public predictor integration, confirmation assets, final figures and the complete empirical paper remain missing. CREATE still needs authenticated access/project location; no unchanged SSH retry or new job was issued here.

No Stage5C, SMC, metric/seconds, true-3D, foundation or new deployment claim. The larger goal remains active and incomplete.
