# Fourfold Training Adequacy: Fitting-Only Results

## Material Passport

Fresh_run: all18 real Torch extended fits, including original-prefix replay.
Cached_verified: original frozen controls. Fresh separate final fitting
diagnostic replay, without new updates. This report reads public scalars.

Actual primary updates: 232,320; first full-refit updates: 8,000.
The three actual seeds and original G-role roster are unchanged. Every
old-budget scientific prefix is exact. Capacity, loss and sampling are
unchanged; compute/exposure are deliberately fourfold, not matched.

## Mean Whole-Fitting Objective

Equal mean across18 original jobs. Known-only is a diagnostic objective
on the same fitted model, not another separately trained arm.

| Objective | Original budget | Fourfold budget | Reduction |
|---|---:|---:|---:|---:|
| bound_aware | 1.326343 | 0.865164 | 34.77% |
| known_only | 0.904304 | 0.543575 | 39.89% |

## Same-Set Calibration

These are median predicted/target mass ratios over444 repeated fitting
recording/pipeline views, not independent-scene inference. Fixed-original
selection holds rows, amplitudes, actual paths and target masses exactly.
Thus any change there is prediction fitting, not less intervention.

| Prediction / selected set | H ratio | R ratio | EH ratio | ER ratio | Actual all/easy violations |
|---|---:|---:|---:|---:|---:|
| extended_fixed_original_selection | 0.325368 | 0.597927 | 0.009333 | 0.038066 | 182/324 |
| extended_own_selection | 0.225213 | 0.652048 | 0.002782 | 0.050984 | 147/307 |
| original | 0.331513 | 0.937621 | 0.003209 | 0.059937 | 182/324 |

Actual violations on the fixed original set cannot change: the paths and
labels are identical. Own-selection fitting risk can change but is not
held generalization, an original complete-risk pass or deployability.
Undefined support stays undefined; no zero-risk substitution.

## Execution And Reproduction

Completion-process PID62172: 570.670s, peak RSS12,068,306,944bytes.
This is the resumed process time, not total experiment wall time. The
initial process stopped at the unchanged disk reserve; its six complete
endpoints and seventh saved optimizer state were resumed unchanged.
The original failed log is retained. Canonical completed updates exclude
any unsaved work lost at interruption (between zero and99 updates).
Separate fitting replay PID63133: 182.927s, peak RSS12,393,955,328bytes.
The full-refit log intentionally repeats step numbers under the same job
name: its checkpoint is first_full_refit.pt.gz, not a budget rollback.

```sh
.venv-pytorch/bin/python scripts/run_m3w_feature_duration_probe.py register
.venv-pytorch/bin/python scripts/run_m3w_feature_duration_probe.py train
.venv-pytorch/bin/python scripts/run_m3w_feature_duration_probe.py verify
.venv-pytorch/bin/python scripts/report_m3w_feature_duration_probe.py
```

Use train --resume for an interrupted same-identity run only. Completed
phases are immutable. Private data/checkpoints are required; scalar
reconstruction is not anonymous cold-start reproduction.

No new H policy scoring or threshold selection. A positive fitting result
requires a separate frozen held evaluation before any generalization
claim. The original feature primary remains inconclusive and risk failed.
SOURCE is exposed development. Pixels/raw frames/silver are not metric,
seconds or human gold. Stage5C execution/SMC remain off; goal remains unmet.
