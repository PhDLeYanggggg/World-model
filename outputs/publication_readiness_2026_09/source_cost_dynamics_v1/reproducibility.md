# Source Trajectory Cost Comparison: Reproduction

## Scope

This is a fixed source-fit diagnostic, not a new test set or a deployed model.
All 22,374 complete stationary-history queries are retained across five admitted
SDD training sites. Training always excludes the held physical site. The 6,460
incomplete stationary queries are not treated as negative labels. Main model
fitting and sealed development, calibration and confirmation roles stay closed.

Registration: `configs/m3w_source_cost_dynamics_v1.json`, SHA256
`857633363c34bb7062c59b6a5e33c769120e7bea9f7c05c7d3c69d714f9a882d`,
committed before fitting in `0007bf43`.

Two losses (ADE and log-ADE), two input arms (mask-only and past RGB), five held
sites and three seeds give 60 fresh models. Every model has 2,000 updates with
batch64. A 100-update pilot belongs to this budget, not an additional fit.
The total is 120,000 updates. Fixed final checkpoints are used; no held-score
checkpoint, architecture or threshold selection is allowed.

The head predicts twelve relative positions, radially bounded to the observed
context radius. Its last layer starts at zero, exactly reproducing stationary
CV. Fifty-one unsupported spatial contexts remain exact baseline predictions.
Both arms receive the same 480 observed geometry/frame features and coverage;
only the RGB arm receives image content. Four matched trials per site/seed share
initialization, row draws, feature normalization and target-loss scale.

## Runtime And Recovery

Use the native arm64 `.venv-pytorch/bin/python`, not x86 Conda/Rosetta. The entry
point checks architecture before Torch use. The recorded environment is Torch
2.12.0, NumPy2.4.6, CPU4 threads, one inter-op thread and zero loader workers.
The runner writes a PID heartbeat and atomic optimizer/RNG checkpoints every
200 updates. The same command resumes; never run two writers in this directory.

```sh
.venv-pytorch/bin/python scripts/run_m3w_source_cost_dynamics.py --registration configs/m3w_source_cost_dynamics_v1.json --audit-only
.venv-pytorch/bin/python scripts/run_m3w_source_cost_dynamics.py --registration configs/m3w_source_cost_dynamics_v1.json
```

Local private artifacts are in
`data/stage_cvpr2027_experiments/source_cost_dynamics_v1/`.
Inspect `heartbeat.json`, `run.log`, `checkpoints/` and `trials/`. Reuse completed
receipts only after their identity, configuration, data and prediction hashes
match. A completed matrix must have 60 receipts and 120,000 recorded updates.

After training has exited successfully, run sequentially:

```sh
.venv-pytorch/bin/python scripts/run_m3w_source_cost_dynamics.py --registration configs/m3w_source_cost_dynamics_v1.json --replay
.venv-pytorch/bin/python scripts/verify_m3w_source_cost_dynamics.py --registration configs/m3w_source_cost_dynamics_v1.json
.venv-pytorch/bin/python scripts/analyze_m3w_source_cost_dynamics.py --registration configs/m3w_source_cost_dynamics_v1.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_cost_dynamics.py tests/test_m3w_source_cost_analysis.py
```

Replay requires bitwise equality of all saved forecast arrays. Verification
checks 15 four-way matched training streams and a completed-run resume with
zero new fits or updates, preserving 181 immutable artifacts and the full report.
The analysis also exports objective loss, normalized batch ADE and logged
pre-clipping gradient norms. Its learning curves are sampled minibatch traces;
they do not establish full-loss convergence. Clipping fractions cover logged
batches only, not every optimizer update.
The plotted trace subtracts same-batch CV, reconstructed from the exact training
sampler seed and weights; its draw counts must match each saved checkpoint.
Public outputs are aggregate CSV/JSON/Markdown and an original scientific SVG;
images, per-row predictions, caches and checkpoints are not committed.

## Interpretation

The main diagnostic is parent-normalized ADE, averaged within physical site and
then equally across sites. Seed losses, not predicted paths, are averaged.
Native-pixel ADE/FDE are separate supplementary measurements. Source +144 raw
frames is not seconds-equivalent to the main native-step observe8/predict12 task.

Compare unguarded forecasts and a fixed probability>=0.9 guard from the frozen
same-arm, same-seed, held-site-excluded classifier. This is an uncalibrated
diagnostic guard, not evidence of safety. The guarded RGB/mask contrast changes
both images and guard decisions; only the unguarded contrast isolates pixels.
Within each arm, ADE versus log-ADE shares the same guard.

Easy examples have exactly zero CV error: percentage degradation is undefined.
Report absolute harm and changed predictions instead of dividing by an epsilon.
Context bounds do not certify physical validity. Binary and containing-ball
oracles use future labels only for diagnostic ceilings, never as input.

Two thousand site-block and within-site video resamples describe conditional
sensitivity. Five repeatedly exposed sites, overlapping training folds and
annotated/interpolated histories do not provide independent confirmation,
strict sensor-as-of observations or human intention gold. No model deployment,
metric/seconds claim, Stage5C execution or SMC follows from this diagnostic.

## Observed Recovery

After a conversation interruption the original PID15300 and session39366 were
both absent. Forty-three completed trial receipts remained. The incomplete
`log_ade_past_rgb_gates_seed29` checkpoint had step600,38,400 sampled draws and
finite weights; all22 registered dependency hashes still matched. The identical
entry point resumed under PID22076/session51870, skipping completed trials.
This is real checkpoint recovery, not a new training budget or a result claim.

## Verified Completion

The resumed trainer exited0 with all60 fits and120,000 updates. All60 checkpoint
forecasts replay exactly. Verification completed15 matched four-way stream
checks and181 unchanged-artifact completed-resume checks with no further updates.
The final report SHA256 is
`6e9abdc5e3e8edaea588a1fa7f37f212578a238dea813f1a33b570bea45a6eb1`.
Fifty-eight focused tests pass; the full legacy suite is not rerun. Figures were
visually inspected. No training or verification process from this run remains
active. Completed numerical execution does not establish forecast utility; see
[the negative result](conclusions.md) and [evidence gates](gates.md).
