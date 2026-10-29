# Reference-Harm Forecast Loss Repair

Prospective registration, 29 October 2026. Status at registration: not_run.

## Question

Recording-balanced policy fitting improves constraint coverage, but the fixed
forecast still lacks a safe easy-reference action in one known source recording.
The original forecaster optimizes mean masked ADE only. Does directly penalizing
its positive excess error produce better candidate forecasts, without changing
the architecture, data, inference inputs or safety budget?

## Intervention And Controls

For each training row, let E be masked forecast ADE, B the ADE of the baseline
selected on that fit's training sources, and C the causal CV ADE. Train with
`mean(source_factor * (E + 2*relu(E-B) + 2*relu(E-C)))`.
Reference errors are training supervision only; future labels never enter input
tokens. This ordinary cost-sensitive loss is not a novel safety theorem.

Reuse all 18 original role definitions: single/complement producer, three
source-locality folds, seeds 17/29/43. Use the same 88,514-parameter Transformer,
motion-bounded output, initial states, source-balanced draws, AdamW, 4,000 updates
per fit and cosine schedule. Do not change the correction bound or shrink output
after seeing scores. Old checkpoints and predictions are cached_verified controls.

Before primary fits, verify 20-update real-data neutral equivalence and 10+10
resume versus 20 direct updates (80 engineering updates in total). Replay one
complete 4,000-update neutral fit and require exact model/optimizer/RNG/draw
equivalence to its original checkpoint. Any failure stops primary training for
diagnosis. Primary budget: 18 fits / 72,000 new updates, not a short probe.

## Data And Evaluation

Reuse the verified European source-training cohort: 318,969 complete-past rows,
163 recordings, 12 localities. Observe 8 steps and request 12 with a raw-frame
stride of 12. Coordinates are image pixels, not verified metric/seconds. Labels
come from released detector tracks, not human gold. All evaluation here is on
previously opened SOURCE development exclusions; it is not independent test.
Selection, calibration and confirmation roles remain unopened.

Freeze all 18 final checkpoint identities in Git before readout. Report both
arms on identical source exclusions and label support, all seeds, all localities
and recordings. Primary contrasts: ADE/FDE versus original forecast, CV and
training-selected strongest; easy and hard cutoffs remain training-only; report
positive-harm ratios, net easy degradation, zero-CV cases, incomplete labels,
worst locality and tails. Report all six causal baselines. Average seed errors,
not trajectories. Use 3,000 paired locality bootstrap draws where estimable;
these conditional exploratory intervals are not independent confirmation.

Recheck the known candidate-support obstruction against the exact frozen policy
floor (a causal CV/damping controller, not the predictor's training-selected
baseline). Reconstruct old eligibility/row alignment; do not silently substitute
a different floor. This label-aware support test is a diagnostic, not inference.
Any inherited candidate-dependent score is stale for a changed candidate and
cannot certify its safety or be deployed without refitting and independent
calibration. Keep all required recordings, including unsupported ones.

## Decision Rules

The 2% full-policy safety requirement is unchanged for every required scene and
recording. Matched counts assess utility only; empty risk is not_estimable.
An average forecasting gain or repaired support does not constitute deployment.
Report any accuracy/harm tradeoff and retain all old failures. No test-selected
weights, seed, checkpoint, threshold or omitted scene. If this loss fails, record
whether it suppressed useful motion, left the floor mismatch unresolved, or did
not fix the unsupported slice before proposing another training change.

## Runtime And Limits

Local native arm64 .venv-pytorch, CPU four compute threads, one interop thread,
num_workers=0. Atomic checkpoint every 200 steps; heartbeat every 50. Resume must
validate objective, code, data and split identities. Check 10 GiB disk reserve.
No new HPC submission is required for the local assets. Weights/caches stay local.
Do not execute Stage5C/SMC. This remains a 2.5D trajectory research experiment,
not true 3D, a foundation model, submission-ready evidence or physical safety.
