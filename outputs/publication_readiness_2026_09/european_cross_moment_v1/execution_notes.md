# Execution and Provenance

Registration commit: `3fe8c81a`, before the real fitting run. Config, model,
runner, model/protocol tests and registration text are bound into immutable
experiment identities. Reporting changes do not alter those fitted identities.

| Phase | Batch mode | Fitting-fixed mode |
|---|---|---|
| Audit and 100-step pilot PID | 59674 | 60247 |
| Resume, full fit, replay, freeze PID | 59793 | 60335 |
| Completed heads | 36 | 36 |
| Completed updates | 72,000 | 72,000 |
| Sum of per-head fitting seconds | 152.134 | 158.343 |
| Unknown-label training draws | 0 | 0 |
| Fixed fitting total loss decreased | 30/36 | 33/36 |
| Fixed fitting rank loss decreased | 25/36 | 28/36 |

Fitting seconds exclude feature assembly, score inference, audits and evaluation.
Native arm64 CPU4/inter-op1/workers0 completed the runs without numerical or
runtime failure. The actual pilots resumed to 2,000 steps/head. Both modes use
identical original minibatches; their 36 fitting-only scale audits match exactly.

Both decision banks were frozen before new evaluation. The last initial decision
was recorded at 2026-09-25 12:16:40 UTC. The final completion receipt additionally
checks this ordering from process events and records the first evaluated group.
Each mode's preceding control is hash-bound and must reproduce exactly.

Fresh work: two sets of real Torch risk heads, checkpoint/sampler replays,
outcome-blind decisions, paired locality evaluation and separate arithmetic.
Cached and verified: original forecast/utility producers, source data, earlier
supported-pair controls and complete source-exclusion chains.
Not run: new trajectory training, independent calibration/confirmation, CREATE
training, or deployment changes. No remote jobs were altered; this comparison
does not claim a fresh CREATE queue inspection.

The full legacy test suite is not run: it contains non-hermetic older paths that
write unrelated artifacts. The relevant registered model, geometry, role,
metric, selection, replay and source-lineage checks are listed in
`completion_checks.json`; the final receipt, rather than this execution note,
establishes their result. Checkpoint binaries and full row-level analyses stay
local. Only code, configs, aggregate tables, figures and reports enter Git.

All twelve localities are opened development. Conditional dependent intervals,
image-pixel 8/12 rawstride12, no independent confirmation or historical Stage37
recertification. No metric/seconds, human-gold, physical-safety, true-3D or
foundation claim. Reserved roles remain closed; Stage5C and SMC remain off.
