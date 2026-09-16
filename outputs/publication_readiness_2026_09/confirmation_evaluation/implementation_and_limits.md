# Frozen Final Evaluation and Seed-Aware Reporting

2026-09-16. `fresh_run` implementation and synthetic training/evaluation checks only. Real clean fitting, development selection, calibration and confirmation remain `not_run`: the scientific protocol is unapproved and independent confirmation data have not been established. No historical external result is restored by this implementation.

## Evaluation Boundary

The final entry reads the comparison family, every training seed, policy settings, baseline identity, recording geometry, query schedule, label rule, error unit and slice definitions from the approved protocol. It has no final-set model ranking or threshold search. Every method family must contain exactly all listed training seeds. Forecaster checkpoint seeds and upstream OOF producer seeds are checked, rather than trusting seed names in a table.

The calibrated deployment policy is read from the actual completed calibration output. The loader verifies its claim, report, completion receipt, source implementation and development-policy lineage. The final comparison uses the same explicit baseline; a nonfloor calibrated producer must be present in the frozen comparison family. A rejected calibration retains the unchanged baseline. Evaluation does not deploy a policy or certify physical safety.

Before reading final labels, the CLI reserves one protocol-bound confirmation claim and binds it to the source implementation and complete execution identity. Resume requires that same family, artifacts, output location and runtime configuration. The API checks the frozen claim; all producers are verified before a confirmation reader is opened. Input decisions precede future-label access. Source/protocol checks are pipeline safeguards, not operating-system restrictions against arbitrary manual file access.

For each frozen predictor and seed, floor, uncontrolled, independent-agent, scene-uniform and joint controls use identical forecasts and agent queries. Equal budget caps do not guarantee equal actual intervention rates, so those rates remain explicit. An explicitly calibrated selected policy is reported separately from the three-seed methodological comparison. It is not an ensemble and must not inherit the latter's replication claim.

## Statistical Estimands

- Prediction errors are reported per seed, with equal-seed mean errors and sample standard deviations. This is not the error of an averaged trajectory ensemble.
- Baseline and label support must match exactly across seeds/families. Past-supported agents with missing future labels remain in decisions and coverage counts; observed ADE/FDE eligibility is reported separately.
- Raw dataset-local ADE/FDE remain per recording. Pooled comparisons require the explicitly approved past-normalized error; no meter or seconds claim is introduced.
- Easy/hard slices use the frozen baseline error definition. Easy preservation is reported separately for every seed, including unsupported/undefined cases, not inferred from a favorable pooled average.
- Positive harm is averaged after taking the positive part for each seed. Taking the positive part after averaging would let good seeds cancel a bad seed's damage; a constructed regression case checks this distinction.
- Bootstrap resamples whole physical scenes, pairing all arms and seeds within the same scene. Training seeds, agents, overlapping windows and bootstrap draws do not increase the independent-scene count. Intervals condition on the fixed trained seeds; seed variability is reported separately, not included as an invented joint uncertainty guarantee.
- Joint-versus-independent, joint-versus-scene-uniform, joint-versus-uncontrolled and joint-versus-floor differences receive paired scene intervals. They are descriptive, not multiplicity-adjusted hypothesis tests or risk certificates. Scene independence still requires substantive evidence.
- Per-scene, per-horizon, tail and worst-scene errors remain explicit. Worst-scene summaries respect the approved weighting; equal-recording evaluation cannot silently use agent-window weights. The aggregate tail is a percentile of seed-mean query errors, with individual seed tails retained separately.
- Joint consistency is the forecast proximity proxy already used by the matched controls. It is not ground-truth collision validation or a physical-safety result.

## Resume and Verification

Per-candidate/recording rows have identity-bound hash receipts. An interrupted recording is recomputed; a completed recording is reused without another prediction/label pass. PID/progress heartbeats, result hashes, completion records and the frozen confirmation claim make recovery explicit. A completed run verifies its result instead of starting a new evaluation. A crash after finishing the claim but before writing the completion marker is recoverable. Changed caches, calibration results, model seeds or execution identity are refused.

20 new cases and related regression checks passed: **177 passed in 62.45 s**. The actual synthetic integration trains nine small forecasters: three seeds, each with two OOF producer fits and one full-fit predictor, eight optimizer updates each. Three ridge risk heads are fitted from their own seed's OOF predictions. Development export, frozen calibration and final comparison then run through the real entry points. These short synthetic fits test the pipeline, not comparative accuracy or scientific convergence.

The inspected synthetic CLI result contains three seeds but only **27 agent queries, 16 scene queries and one physical scene**, with 15 complete ADE/FDE labels. It correctly reports no scene-bootstrap interval and retains the calibrated floor. No synthetic improvement percentage is promoted to the project's results. A separate constructed two-scene table checks the paired-bootstrap calculation, including seed variation and the harm-cancellation counterexample.

A regression run exposed a NumPy Boolean serialization error in the new per-seed easy check. Converting the result to a native JSON Boolean repaired it; strict JSON serialization and complete CLI recovery subsequently passed. The failure is implementation evidence, not a neural-performance failure. Previous full-suite status remains 1,870 pass / 1 unrelated fail; that nonisolated suite was not rerun.

Real preflight still exits 2 with `Explicit protocol approval required`, before training or final-label evaluation. No current real forecasting accuracy, formal three-seed result, confirmation CI or deployment gain has been produced. See [verification](verification.json), [tests](../../../tests/test_m3w_confirmation_evaluation.py), and the [operator runbook](../local_create_runbook_zh.md).

## Remaining Work

The implementation now connects clean forecasting, cross-fitted costs, development selection, frozen risk screening and final-family reporting. Its availability does not supply the missing scientific choices or independent data. The primary task, data roles, aggregation and risk tolerances still await explicit approval; the current six historically exposed scene groups cannot simply be renamed into untouched confirmation data. CREATE access remains unresolved, with no new HPC job submitted. After approval, the next substantive evidence is matched real fitting and development comparison, followed by appropriately independent confirmation, not another synthetic performance claim.

Stage5C and SMC remain disabled. The project remains dataset-local/raw-frame, not verified metric, seconds-level, true 3D or foundation modeling. CVPR submission readiness is not achieved.
