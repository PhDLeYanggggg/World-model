# Baseline-Relative Joint Intervention: Implemented Prototype

## Evidence Status

2026-09-16. `fresh_run` engineering checks on synthetic score problems and `cached_verified` raw-position caches. No new forecasting model was trained. No real predictive gain, independent risk guarantee, deployment promotion or submission readiness is established. The primary experiment protocol remains awaiting user approval.

Previous goal turn: substantive progress, with a committed causal reader and verified arm64 training runtime. This turn implements the decision mechanism and tests it, rather than treating a proposed architecture as evidence.

## Implemented Interface

`src/world_model/m3w_joint_intervention.py` separates inference from loss/evaluation labels:

- Inference receives a fixed baseline forecast, a fixed candidate forecast, predicted gain/harm, past-defined support and edges. It never receives future targets, future availability, oracle labels or evaluation easy labels.
- `realized_relative_costs` creates signed ADE/FDE gain and positive harm labels using the same valid target mask for both forecasts. Normalization scales are caller-supplied past scales. Missing final endpoints remain missing; the last available earlier point is not relabelled FDE at the requested horizon.
- Five controls share the candidate forecasts: floor, uncontrolled supported candidate, unary/independent selection, whole-scene selection and joint selection. The uncontrolled arm is explicitly not required to satisfy the intervention budgets.
- The independent arm solves the unary budget-constrained problem without pair compatibility. It is a stronger comparator than a naive per-row threshold. A shared cardinality cap is not evidence of matched realized intervention rate; the demonstration separately checks an equal-cardinality comparison.

The reader now exposes a per-agent reversible transform determined only by observed history. Predictions are restored to shared dataset-local coordinates before pair comparison. Agent IDs and prediction grids must align exactly; silent interpolation and dropping agents without future labels are prohibited.

## Objective and Solver

For n observed agents, let g_i predict expected baseline loss minus candidate loss, and h_i predict the expected positive excess loss of the candidate. Unsupported agents stay on the baseline but remain in the scene mean. Both heads must use the same declared loss units. Reconcile independent heads using h_i' = max(h_i, -g_i, 0), since expected positive harm cannot be smaller than negative expected gain.

The binary prototype minimizes

```text
-(1/n) sum_i g_i x_i + (lambda / |E|) sum_(i,j) C_ij(x_i, x_j)

subject to:
    (1/n) sum_i h_i' x_i <= rho
    sum_i x_i <= intervention_cap
    x_i <= supported_i
```

When E is empty, the pair term is zero. C_ij is the positive increase in a candidate-trajectory proximity proxy relative to the all-baseline pair; C_ij(0,0)=0. The proxy averages squared threshold penetrations at declared prediction timestamps. It is not a continuous-time collision check, does not know agent body extent, and does not establish physical safety. The graph radius, distance threshold, lambda, rho and intervention cap require explicit inputs; no paper risk budget has been chosen.

The product y_ij = x_i x_j is linearized with y<=x_i, y<=x_j, x_i+x_j-y<=1. SciPy/HiGHS solves the resulting MILP. A non-optimal solve, failed post-check or nonpositive predicted benefit returns the baseline. This guarantees only behavior of the optimizer under supplied estimates, not realized predictive safety. See the [SciPy solver contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).

## Fresh Checks

| Check | Result | Interpretation |
| --- | --- | --- |
| MILP versus exhaustive enumeration | 80 synthetic seven-agent problems; maximum objective difference 0 | Correctness on these finite problems, not learned accuracy |
| Runtime on those problems | Median 0.00651 s; maximum 0.00909 s | Small synthetic workloads only; no dense-scene latency guarantee |
| Common-frame reconstruction | 36 real scene queries, 497 agent queries, seven causal baselines each | No future labels read; maximum absolute error 2.53e-6 dataset-local units |
| Updated causal reader audit | 142,402 indexed views; 144 window and 36 scene corruption checks pass | Coordinate-transform metadata also invariant to hidden future coordinates |
| Focused regression suite | 48 passed in 2.37 s | Includes latest solver, geometry, lineage, selection and isolated legacy test |

After the reversible-coordinate refactor, the 1,024 engineering runtime inputs were regenerated without Torch or future labels. Their float32 input/target byte hash still equals `32425db4160f0b141f6a13927a8badd63dc16a40233628fbfa9d5e9c5356d849`. Prior CPU/MPS probe evidence is reused for identical arrays, not presented as a fresh runtime measurement.

The synthetic example deliberately exposes a tradeoff. At the same 2/3 intervention rate, independent selection has greater predicted unary gain but creates the constructed pair conflict; joint selection reduces that proxy at lower predicted unary gain. It does **not** establish better ADE/FDE. The first example used too small a pair coefficient to produce its intended optimum; the hand-designed fixture was corrected analytically, not tuned against real test data.

No full-suite rerun was needed for unchanged legacy modules. The preceding run remains **1,870 pass / 1 unrelated data-lake fixture failure**, not a green suite. Its non-hermetic report writes are documented in `../verification_record.md`.

## Calibration Primitive and Limits

`screen_cluster_risks` implements a conservative Hoeffding plus union-bound screen for a fixed family and multiple bounded mean risks. It rejects repeated calibration cluster IDs, overlap with declared fitted/development IDs, missing data and values outside supplied loss bounds. It does not silently clip unbounded ADE into a different statistical target.

The screen cannot verify that user-supplied cluster IDs are independent, that the family was fixed before calibration, or that a chosen loss bound is scientifically meaningful. Returned flags explicitly leave those assumptions unverified. It is not a conformal or physical safety certificate.

An illustrative calculation uses 64 fixed policies, two losses bounded in [0,1], delta=0.05 and tolerance=0.02, with every empirical loss set to zero. Six independent clusters give an upper bound of 0.809 and no passing policy. Ten thousand give 0.0198. These are synthetic inputs, not actual calibration outcomes, and **0.02 here is not the project's easy-degradation ratio**. The conservative calculation is not a universal lower bound on the sample size needed by all methods. With six current physical groups, many of which would also be needed for fitting/development, a useful independent-scene guarantee is not established.

## Reproduction

```bash
.venv-pytorch/bin/python scripts/check_m3w_joint_intervention.py
.venv-pytorch/bin/python scripts/audit_m3w_causal_recordings.py --report-dir outputs/publication_readiness_2026_09/joint_intervention
.venv-pytorch/bin/python -m pytest tests/test_m3w_joint_intervention.py tests/test_m3w_causal_recordings.py tests/test_m3w_recording_lineage.py tests/test_stage44_worldcore.py tests/test_stage42_source_level_ucy_full_waypoint_integration.py -q
```

Machine evidence: `mechanism_checks.json` and `causal_recording_checks.json`. Synthetic scores and budgets are fixture constants, not fitted hyperparameters. No official split, holdout exposure, new teacher, Stage5C or SMC is introduced.

## Next Scientific Step

Approve the main temporal/coordinate protocol and fitting/development/calibration/confirmation roles, then train a fixed predictor and cross-fitted gain/harm heads. Compare these control arms on the same forecasts across predeclared risk/coverage settings, including error, easy degradation, tails, worst scene and geometric consistency. Keep physical-scene clusters intact. Do not present this implemented optimization mechanism as a novel or empirically successful method before that comparison.
