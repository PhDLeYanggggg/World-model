# Fixed Capacity and Duration Comparison

## Material Passport

Research-development code experiment, using four already exposed SDD source
sites. Motivation: the fixed intermediate loss repairs high-disagreement ranking
but loses lower-risk opportunities. This is not independent confirmation.

## Hypothesis and Matrix

With causal disagreement D in X, positive weighted squared-error objectives
have the same unrestricted conditional-mean optimum. Their observed tradeoff
may therefore reflect limited fitting capacity or incomplete optimization,
rather than solely a missing signal. A finite factorial can test these two
changes, but cannot prove that more training would never help.

Fix exponent-one loss, features, normalization, optimizer, learning rate,
sampling, forecasts and all policy thresholds. Compare hidden widths 64/128
and update endpoints 3,000/12,000 at seeds 17/29/43 and four outer sites.
Retain the same bounded two-cost architecture; only hidden width changes.
Original 64-wide/3,000-update heads are cached-verified. Continue their exact
model, optimizer and sampler states to 12,000 in new output folders. Train
twelve 128-wide paths from the same registered seeds, retaining their 3,000
and 12,000 endpoints. No optimizer reset or learned stopping rule.

This yields 12 new wide training paths and 12 narrow continuations, not 36 new
independent fits. There are 36 newly computed endpoints, 252,000 new updates
and 64,512,000 new draws. Inherited draws/compute are reported separately.
The first 3,000 wide draws must equal the existing narrow draws; the complete
12,000 narrow/wide draw vectors must agree. A prefix is not a completed matrix.
Copying metadata into a new continuation contract must not alter any parent
checkpoint, model, optimizer or RNG state. All old artifacts remain frozen.

## Fixed Readout

Primary: wide/12,000 strict-stop minus the existing refitted fraction-strict
equal-site native ADE gain. Do not use the weaker intermediate-loss result as
the only primary comparator. Require lower paired 3,000-scene-bootstrap bound
above zero, positive gain for every seed, each seed's aggregate easy degradation
at most 2%, and zero complete exact-CV harms. Report per-site/seed easy values
even when aggregate checks pass. Independent safety is not inferred.

Report all four arms and net-stop, strict-stop and matched-count policies.
The count anchor remains the original frozen transferred fraction-strict rule.
Report fixed capacity/duration contrasts, including their interaction, without
winner selection or promotion of a favorable secondary. Also report FDE,
hard/easy, tails, worst scene, coverage, unknown/incomplete outcomes and full-grid
gain bounds. Existing fitting-q50/q90/q99 disagreement/speed diagnostics and
training loss traces quantify where fitting changes. Fitting metrics are
optimistic; held-source improvements are not independent confirmation.

Freeze decisions before aggregate outcome readout. No outcomes or future support
enter inference. No new original val/test/main/external/bookstore role is opened.
No normalization/goal leakage, new calibration, threshold sweep or risk allowance.

## Runtime and Limits

Native arm64, CPU4/inter-op1/workers0, process lock, atomic 500-update checkpoints
and 100-update heartbeat. Resume explicitly; never restart because a poll expires.
Perform a short pilot to estimate cost, then run the full fixed matrix. Use local
cached heads; no new CREATE workload is needed unless resource evidence changes.
Replay every endpoint and verify sampling/choices/metrics with separate code.

Eight observed/twelve predicted sampled annotation steps, SDD stride12,
annotation-pixel ADE/FDE. No metric/seconds, true3D, foundation, deployment,
Stage5C, SMC or submission-ready claim. The matrix is an optimization diagnosis,
not a novel method simply because one setting performs better.
