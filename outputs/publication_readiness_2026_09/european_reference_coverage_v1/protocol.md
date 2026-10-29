# Causal Reference-Coverage Comparison

## Material Passport

Experiment execution, prospective matched ablation, 29 October 2026. Scope:
already-opened European SOURCE development. Not an independent test or a
submission-ready result. Training and readout are not_run at registration.

## Hypothesis and Single Changed Factor

The last fixed loss comparison reduced some positive harm but lost predictive
accuracy. A geometric check found that the fold-2 correction ball could not
represent causal CV on 43,743 excluded-source windows. That limitation did not
explain the remaining unsupported 082/51 view, which is retained.

Test whether completing reference coverage improves the trained forecaster:

`B_new(t) = max(B_old(t), 1.05*||CV(t)-center(t)||, 1.05*||D97(t)-center(t)||)`

`prediction(t) = center(t) + B_new(t)*radial_squash(delta(t))`

The center remains each producer's training-selected baseline. CV and damping
.97 are computed from past history only; .97 is included because the frozen
policy's causal floor chooses between these two trajectories. The 5% margin
puts nonzero reference displacements strictly inside the open radial ball.
Zero-motion fixed references remain exact. No future labels, teacher or learned
calibration determines the radius. This is representation coverage, not a
guarantee that training chooses a reference or that predictions are safe.

No parameter, feature, split, baseline center, train-source scale, loss weight,
sampling, initialization, AdamW setting or 4,000-update schedule changes. The
primary comparator is the frozen harm-aware producer from the preceding turn.
The original ADE-only producer is a secondary frozen comparator, not an arm
retrained here. No threshold or margin search, favorable-seed selection or
checkpoint selection. All nine endpoints must be frozen before readout.

## Execution

Nine single-fold producers, folds 0/1/2 and seeds17/29/43: 36,000 primary updates.
First run a real-data 80-update pilot on fold2, requiring exact coverage-off
versus control and exact 10+10 versus20 resume. Then a full4,000-step neutral
replay on fold2 against its preceding harm-aware checkpoint. Validate identical
parameter keys/initial weights and draw/RNG state. When both train and evaluated
radii are unchanged, require exact final scientific-state replay, not just
similar performance. Report any such neutral fits separately from affected fits.

Preflight coverage uses only past geometry on the already-opened source
population. Report train/excluded affected counts and CV/D97 outside counts
before/after. No selection/calibration/confirmation roles are opened.

Use native arm64 .venv-pytorch, CPU4/interop1, workers0, checkpoint200 and
heartbeat50 updates. Local assets already exist and comparable runs fit in
under2GB process memory, so use local training after a measured pilot. Preserve
10GiB free disk. No new HPC training is needed. Recovery resumes the same atomic
checkpoint and budget; healthy slow execution is not a failure.

## Fixed Readout and Decision

Same 318,969 source windows,163recordings,12localities, eight observations and
twelve requested predictions, stride12raw frames. Same masks, complete-future
subset, train-defined easy/hard cuts and six causal baselines. Report ADE/FDE,
all/easy/hard, zero-reference and unknown-label support, per-seed, per-locality,
per-record tails and positive harm, with3,000paired locality-bootstrap draws.
Average repeated producer/seed errors within locality, not trajectories. Twelve
localities, not overlapping windows or repeated contexts, are uncertainty units.

Primary paired contrast: new versus harm-aware control, equal-locality ADE in
CV-relative percentage points. Secondary: new versus original ADE-only model.
Report all results, including any safety/accuracy tradeoff, rather than declaring
a win from one positive slice. A positive source-development result is not an
independent generalization or deployment claim.

Reproduce the original148-view TRAIN fixed-floor diagnostic, then replace only
candidate trajectories and their labels/envelopes. Keep the exact CV/damping.97
floor, original eligibility, source/record roster and2%constraints. Report all
formerly unsupported views and any new loss of support. This oracle is
label-aware and not an inference policy. Old candidate-dependent learned costs
are invalid for changed forecasts; no deployment without new costs and
independent calibration.

The complete-policy2%all/easy requirements on every required recording remain
unchanged. Net error cannot cancel positive harm. Empty matched risk remains
not_estimable; count matching is utility-only. Historical failures remain.

Only consider a follow-up policy refit if accuracy, harm and support provide a
reason to do it. Otherwise preserve the existing deployment and diagnose the
failure. Do not promote an oracle or unchanged neutral seeds as new capability.
Image pixels/raw frames and detector-track labels only: no metric, seconds,
human-gold, true3D, foundation, Stage5C execution or SMC claim.
