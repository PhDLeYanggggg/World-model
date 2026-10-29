# Earlier-Family Matched Loss Result

Fresh_run: all 18 neural fits, 72,000 primary updates and all fixed readouts
completed. Cached_verified: original source geometry, controls and predictions.
Training PID35068 exited 0 after 1,640.711 seconds, with 1.734 GB peak RSS.
Readout PID38791 exited 0 after 486.671 seconds. Replay PID40091 exited 0
after 29.260 seconds and reproduces the analysis and per-record CSV exactly.

The experiment adds fixed coefficient-2 positive-harm penalties against the
training-selected baseline and causal CV, with the original architecture,
folds, initialization, sampling, optimizer and 4,000-update budget. It is the
earlier flat complete-neighbor forecaster, not the current policy producer.
The lineage correction is preserved in `lineage_clarification.md`.

All 18 endpoints were committed in `41f0d3f0` before first readout. The primary
complement-fold analysis includes each source row once per seed, averaging
errors across seeds rather than ensembling trajectories.

| Equal-locality endpoint | Original (%) | Harm-aware (%) |
|---|---:|---:|
| ADE gain over CV | 2.157596 | 1.389049 |
| ADE gain over training-selected baseline | 4.113500 | 3.535102 |
| FDE gain over training-selected baseline | 5.221954 | 4.620432 |
| Hard ADE gain over CV | 8.073580 | 7.452622 |
| Easy ADE gain over CV | -13.924449 | -13.552605 |

The paired all-ADE change is -0.768547 percentage points, with a 3,000-draw
conditional locality-bootstrap interval [-2.281846, +0.832023]. There is no
stable positive lift. The interval spans zero, so this is not a broad claim
that harm-aware training is always worse.

The compact `readout.md` includes pooled easy net-error and positive-harm
columns. Those are descriptive aggregates, not the equal-locality primary
easy metric above and not full-policy safety. In particular, pooled easy
improvement cannot be used to waive per-record risk.

The matched neutral replay, exact interrupted resume and original primary
metric replay pass. These validate implementation, not the research hypothesis.
No independent role was opened. The data are 318,969 already-opened SOURCE
windows, 163 recordings and twelve localities, with 7,047 entirely unknown
future-label rows retained. Overlapping windows are not independent samples.

This family does not repair the deployed policy. Do not promote its loss,
choose a favorable seed, or mix its results with the later producer. Complete
policy safety still requires the original 2% conditions for all required
recordings, with empty matched risk not_estimable and matching used for utility
only. Historical failures, deployment, Stage5C and SMC remain unchanged.
Coordinates remain image pixels and horizons raw frames, not metric/seconds,
human gold, true 3D or foundation evidence.
