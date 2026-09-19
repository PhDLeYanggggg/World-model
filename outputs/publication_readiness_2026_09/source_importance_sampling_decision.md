# Fixed Importance-Corrected Episode Sampling

## Question And Reason

The completed equal-episode experiment changed the expected training risk.
Every new model fits its reweighted training task better than stationary CV,
while worsening the original unweighted risk. It fails out of site as well.
The next falsifiable check restores the uniform-row expected ADE while retaining
the same episode exposure. This tests whether the measured objective shift
explains the large harm increase. It does not promise transferable dynamics.

## Frozen Scientific Contract

Use the same 15,430 source-only stationary-history queries, four explored site
folds (coupa/deathCircle/gates/hyang), seeds 17/29/43, geometry and centered-image
arms, 63,960 parameters, zero-initialized bounded output, 10,000 updates, batch64,
AdamW, original learning-rate schedule, normalization, training cost scale and
hard cutoff. Same offline eight observed/twelve predicted annotation steps,
stride12 raw frames. Keep all static and moving-target rows in evaluation.
No main, bookstore, outer, independent confirmation or external readout.

There are exactly 24 fresh heads/240,000 updates. The named coupa geometry seed17
100-update pilot is included, saves a checkpoint and emits no forecast.
No data-driven early stopping, model/seed/checkpoint selection or threshold sweep.
The original 24 uniform geometry/centered fits and 24 uncorrected episode fits
are hash-verified cached controls, not fresh retraining.

## Single Intervention

Reuse past-defined annotation groups. Inside each training complement, sample
each group equally and each of its rows equally. Keep the same weighted draw
sequence as the uncorrected control for the matching seed/site/arm.

For normalized probability p_i and N training rows, multiply each sampled
per-row ADE by w_i=1/(N*p_i), then average over the fixed batch size. Do not divide
by the sampled weight sum, clip factors or fit them using held labels/frequencies.
Every row has positive probability. All targets remain loss/evaluation only.

Check sum_i p_i*w_i*loss_i equals the uniform loss, and the corresponding
unclipped gradient identity, on exhaustive small batches and actual training
labels with a fixed shared-offset probe. Check uniform sampling recovers the old
trainer exactly, interrupted resume is exact and changed propensities are rejected.
Gradient clipping at5 and AdamW stay unchanged. The unbiased identity applies to
loss/unclipped gradient, not the nonlinear optimizer update. Log clipping and
weight moments; increased stochastic variance is a possible failure mechanism.

## Fixed Evaluation

Primary: original ratio of equal-site mean normalized ADE against stationary CV.
Mean seed errors, not a prediction ensemble. Retain all six arms, all sites/seeds,
native annotation-pixel ADE/FDE, hard-slice gains, zero-target absolute harm,
nonzero-target gain, p95/p99 and binary oracle diagnostics. Zero-CV percentage
easy degradation is undefined; do not call it a 2% pass.

Use the parent's 2,000 shared four-site bootstrap draws and recording-level
descriptions. They are conditional on explored sites/shared training folds, not
independent confirmation. Fixed contrasts: each corrected arm versus uniform,
each corrected arm versus uncorrected episode sampling, and centered-corrected
versus geometry-corrected. Harm reduction relative to a failing control is not
positive prediction benefit. Training-risk analysis is diagnostic, not selection.

## Runtime, Artifacts And Boundaries

Native arm64 .venv-pytorch, CPU4/inter-op1, workers0. Prior comparable 240k updates
took531seconds locally;64GiB free at setup. Recheck the included pilot before
full fitting. Save atomic checkpoints every200steps and heartbeat with PID.
CREATE historical SSH failure is not a fresh remote-state check; no job submitted.
This local experiment is feasible without moving private caches or probing GPUs.

Register code/config/test hashes and push before fitting. Exact forecast replay,
regenerated sampler draws, matched old-control draws, real factor checks, OOF
recomputation and zero-update completed resume are required. Retain failures.
No raw data, image/feature/history caches, weights or checkpoints in Git.

Supplied past annotations can use later interpolation controls: offline, not
sensor-as-of. No metric/seconds/true3D/foundation or independent-publication claim.
No new policy/deployment, Stage5C execution or SMC. A successful mathematical
correction or lower harm alone cannot satisfy the world-model research goal.
