# Source Trajectory Cost Evidence Gates

Engineering completion is separate from research success. This diagnostic does
not replace the main task's gates or restore historical exploratory test scores
to independent confirmation.

| Requirement | Status | Evidence and scope |
| --- | --- | --- |
| Registered fixed design | Pass | Commit0007bf43; unchanged registration and22 dependency hashes |
| Approved data roles | Pass within audited protocol | Original SDD train40 only; no main fitting or sealed scoring |
| Past-only inputs | Pass within offline-annotation contract | Future targets only in training loss/evaluation; no test endpoint goals or central velocity; offline supplied annotations are not sensor-as-of |
| Full fixed training budget | Pass | 60 models,120000 updates, three seeds, five source sites; not a partial pilot |
| Recovery and reproducibility | Pass | Actual step600 recovery;60 exact replays;15 matched four-way streams;181 artifacts unchanged on completed resume |
| Numerical execution | Pass | Finite outputs/training and context bounds;58 focused tests; no convergence claim |
| Direct cost alignment improves forecasting | Fail | All uncontrolled objective/input summaries and all60 fits worsen held-site CV ADE |
| Incremental RGB contribution | Not established | Both uncontrolled matched RGB contrasts have intervals crossing zero |
| Training utility | Fail | All60 complete training-set ADE scores worse than CV |
| Easy preservation | Not established | Zero baseline denominator; positive absolute harm; no invented percentage pass |
| Risk-calibrated guard | Not run | Frozen0.9 classifier gate is diagnostic, not independently calibrated safety |
| Statistical sensitivity | Complete, limited | 2000 conditional site/video resamples; exposed sites and overlapping training folds |
| Independent confirmation | Not run | Main development/calibration/confirmation remain sealed |
| Main forecast or joint-intervention contribution | Not established | Source stationary diagnostic does not supply this evidence |
| Deployment upgrade | False | Retain prior deployment boundary; no new model promoted |
| Stage5C execution / SMC | False / false | Neither is enabled |

Verdict: `complete_negative_source_trajectory_cost_comparison`.
Project submission goal remains unmet, not blocked solely by this negative result.
