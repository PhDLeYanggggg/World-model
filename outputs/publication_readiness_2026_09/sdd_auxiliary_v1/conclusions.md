# SDD Auxiliary Transfer: Complete Matched Results

Result source: fresh real Torch training; source and main caches hash-verified.
All 54 preregistered fits and 324,000 updates completed. No held-fit checkpoint selection.
This is exploration on three previously exposed physical sites, not independent confirmation.

| Schedule / input | Gain vs CV (%) | Descriptive site CI (%) | Gain vs same-input no-SDD (%) | Pixel gain vs mask (%) | Safe positive fits |
| --- | ---: | --- | ---: | ---: | ---: |
| no_aux_geometry | -1.34976 | [-12.88697, -0.16553] | 0.00000 | 0.02236 | 0/9 |
| no_aux_mask_only | -1.37243 | [-12.87146, -0.15183] | 0.00000 | 0.00000 | 0/9 |
| no_aux_past_rgb | -1.98266 | [-9.76082, -1.23619] | 0.00000 | -0.60197 | 0/9 |
| sdd_aux_geometry | -0.80523 | [-7.22531, -0.22560] | 0.53728 | 0.01094 | 0/9 |
| sdd_aux_mask_only | -0.81626 | [-7.32295, -0.21206] | 0.54864 | 0.00000 | 0/9 |
| sdd_aux_past_rgb | -1.27335 | [-6.79648, -0.78637] | 0.69552 | -0.45339 | 0/9 |

## Interpretation Boundaries

Positive source transfer versus a neural control is not necessarily improvement over causal CV.
RGB contribution is judged against the same-source spatial mask arm, not only geometry.
A bootstrap over three reused sites cannot create independent scene evidence or certify safety.
Easy relative degradation is retained alongside absolute normalized harm and all seed/site failures.
No model is promoted by this fit-only comparison. Stage5C and SMC remain disabled.

## Source and Sampling

Original train-only videos: 40; full eligible auxiliary population: 229,333.
Complete/partial/absent future labels: 188,358/37,360/3,615.
All windows are indexed without future-support selection. Uniform sampling with replacement
uses 128,000 auxiliary draws per auxiliary fit, not one full epoch. Unique row counts are in CSV/JSON.
Zero-label rows stay in the population but are excluded from supported-loss averaging.
SDD uses stride 12 raw frames, 8 past/12 future points, endpoint +144 raw frames; no seconds equivalence.
Main task keeps its original annotation steps, complete labels and physical-scene-equal primary.
Source histories include retrospective annotation interpolation. No strict sensor-as-of claim.
No metric, true 3D, foundation, generalization-success or independent-test claim.

## Remaining Uncertainty

The source contrast jointly changes dataset, viewing geometry, motion distribution and label support.
It does not identify any one of these as the cause of a gain or failure.
Main-train-only normalization is fixed in both arms; source clipping is reported, not tuned away.
Low-resolution crops and domain mismatch remain hypotheses, not causal findings.
For a publishable gain, a future protocol needs a stable method effect and untouched scenes.

## Reproduction

Run with native arm64 .venv-pytorch, four Torch threads, one interop thread, no workers.
Use the registered configuration with prepare_m3w_sdd_auxiliary.py,
verify_m3w_sdd_auxiliary_inputs.py, run_m3w_sdd_auxiliary.py, then its --replay mode.
Rerunning the completed trainer verifies artifacts and adds zero optimizer updates.
Private arrays/checkpoints are not in Git; hashes, registration, code and aggregate results are.
See [fit metrics](fit_metrics.csv) for all 54 fits and [analysis](analysis.json) for event/native-coordinate diagnostics.
The [failure analysis](failure_analysis.md) retains easy harm and competing explanations.
See [reproducibility](reproducibility.md) for exact replays, resume and test scope.
