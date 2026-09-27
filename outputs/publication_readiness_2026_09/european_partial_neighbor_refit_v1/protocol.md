# Matched Partial-Neighbor Neural Refit

Source-development experiment, registered before new neural fitting/readout.
The motivating input audit is already observed. This is not an untouched test.

## Hypothesis
The complete-history neighbor restriction removes current-visible interaction
information. Keeping valid partial-neighbor tokens may improve forecast dynamics
under an otherwise matched training procedure. Coverage alone is not lift.

## Fixed Design
Reuse the original 12 source localities, three four-locality producer folds,
seeds17/29/43, all 318,969 past-eligible queries and the existing raw future
labels. Train nine new full-context forecasters for 4,000 updates each. Use
the original width64/head4/layer2 Transformer, batch64, AdamW schedule, training-
locality baseline choice and loss weighting. No outcome-based row filtering.
The source task stays obs8/pred12 at raw-frame stride12. No time/metric claim.

The only new mechanism is the versioned context-support rule: up to eight
nearest current-visible neighbors, valid per-step masks, observed-token
attention and scaling over observed tokens. Ego history, baseline outputs,
parameter count, zero-initial correction and motion-bounded output are unchanged.
Do not combine this with smoothing, another loss, architecture or threshold sweep.

Nine old four-locality forecasters are cached controls. Before accepting them,
freshly reproduce the first legacy4000-update endpoint and check every model
parameter and sampling-count vector exactly. Other controls are hash-verified.
Compare initialization and full sampling sequences for every new/old pair.
New weights and caches use a separate directory. A100 is available only through
the approved scheduler; the bounded local pilot determines whether moving is
justified. CPU arm64, four threads, zero workers; atomic checkpoints every200
updates and heartbeats every50, explicit resume, >=10GiB free-space reserve.

## Separation and Freezing
Every prediction locality is excluded from that predictor's fitting and
preprocessing. Keep the producer/controller/readout assignment lineage for
later intervention work. Here only forecasting changes: no cost-head fitting,
calibration, policy search or deployment. Independent selection, calibration and
confirmation stay closed. Freeze all nine new and nine matched control
prediction banks in Git before comparative source-outcome scoring. Target
fields and future-valid masks never enter prediction inputs.

## Endpoints
Primary: equal-locality mean percentage ADE gain against matched old neural
forecasts. Average the two producer contexts and three seeds inside each of
12 source localities, then use3,000 locality bootstrap resamples. Report per-
seed and per-producer results, FDE, all/positive-easy/hard/zero-CV subsets,
training-selected causal baseline and CV comparisons, positive gain/harm,
tail error, recording and locality counts, request smoothness and finite output.
Easy/hard cuts come from each predictor's training design. Zero references
remain undefined percentages with absolute costs, never epsilon gains.

An exploratory benefit requires a positive primary lower interval, positive-
easy degradation no greater than2% against the matched old neural forecast,
and no negative hard-gain point estimate. Report positive-easy costs against
the causal baseline too. These are forecast screens, not calibrated policy
safety or independent deployment certification. All source outcomes were
development-exposed and producer fits overlap; bootstrap is conditional.
No result is selected by seed, locality, epoch or threshold.

## Interpretation
If this mechanism fails, retain the failed contrasts and diagnose tail/easy
costs and partial-neighbor sensitivity. Do not keep sweeping the same cap.
If it succeeds, the next required evidence is a matched gain/harm head and
scene-joint intervention with preserved exclusions and independent calibration.
Never claim risk-policy benefit merely from raw forecaster gain. No SDD rerun
is implied. Stage5C execution and SMC remain disabled.
