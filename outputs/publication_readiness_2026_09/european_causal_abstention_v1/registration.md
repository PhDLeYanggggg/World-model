# Causal Abstention With Same-Frame Count Controls

Registered before new decision outcomes, 25 September 2026. Development only.
The preceding floor-relative target experiment failed its matched repair test.
Its frozen summary hash is bound by the configuration; no preceding result is
being erased. Post-hoc inspection found four zero-CV rows with a stationary last
observed step and earlier movement. Those labels have only two future points,
no endpoint, and no zero-CV fitting support in the two evaluating folds. This
motivates a new exploratory hypothesis, not an independent confirmatory test.

## Hypothesis and Fixed Matrix

Does causal stop/support abstention improve which neural interventions survive,
beyond simply reducing the number of interventions?

Keep the neural forecaster, source-excluded protected damping/CV floor, fitted
utility/risk heads, all 380 original features, label population and budget fixed.
Use both CV-target and both-floor-target parent endpoints; intermediate factorial
arms were tested previously, not silently optimized or discarded here. Two
normalizer modes, three source folds, seeds 17/29/43 and all/easy events give
36 dependent views per policy, not independent tests.

Each parent has ten policies: original; stop, support, combined; plus risk-ranked
and deterministic-random count-matched controls for each of the three guards.
Total: 720 views. Report every view. No model refit or threshold search.

Stop requires positive latest observed displacement. Support uses three causal
quantities: latest/mean past-step speed, mean turn angle over nonzero adjacent
past steps, and mean neural-to-floor rollout disagreement divided by twelve
times mean past-step speed. Exact zero denominators yield zero features and a
separate always-stationary state. Other states: terminally stopped and moving.

Fit 1st/99th percentile axis-aligned boxes for each fitting locality and motion
state, using known-label FITTING rows only; require at least 32 such rows per box.
An inference row must be inside boxes from at least two distinct fitting
localities. Combined requires stop AND support. These fixed engineering values
are not estimated from held outcomes. Overlapping rows are not independent
support samples. No claim that rectangular support implies distribution coverage.

For each guard, retain exactly its number of interventions in each recording
and observed current frame. Reallocate among the parent's original interventions
using ascending predicted H/B, or SHA256(seed:row_id) order; ties use row_id.
All alternatives are subsets of the original policy. Queries contain the indexed
eligible eight-step histories, not a claim to include every visible agent.
No count is pooled across future frames or held localities. Risk controls obey
the original predicted risk limit but may select support-rejected rows, which is
intentional for the matched comparison. No forecast is suppressed: abstention
means retaining the protected floor.

## Readout and Integrity

Both complete decision banks must freeze before either new readout. Recompute
every guard/ranking with a separate scalar implementation. Verify source exclusion,
hashes, prior decision heads, exact count equality within query and original
metric replay. Compare coordinate errors with separate arithmetic and every
aggregate with independent locality reduction.

Report all/easy/hard/complete ADE relative to protected floor and original parent;
endpoint FDE, positive-CV easy degradation, zero-CV harm, unknown-label switches,
tail/worst-locality error and intervention rate. Report guard-versus-count-matched
controls and removed-benefit/avoided-harm decomposition under a common per-locality
floor denominator. Paired locality bootstrap: 3,000 draws, seed 39271, same existing
eight-locality roster per fold. Overlapping folds/seeds remain dependent.

Evidence for support-aware ranking requires benefit over count-matched controls,
not only fewer harmed points. Zero-CV preservation alone is insufficient; disclose
loss of useful interventions and hard/complete-label outcomes. No variant is
promoted using these opened development results. There is no predeclared deployment
selection here. Independent model-selection, calibration and confirmation roles
remain closed; this is not conformal risk control or a physical safety guarantee.

## Execution and Limits

Local native arm64 Python, CPU four threads, one inter-op thread, no DataLoader
workers. No new neural training. Per-group immutable decisions/results, heartbeat,
lock, resume and 10-GiB free-disk guard. CREATE read-only state checked separately;
no HPC job required for this fixed-forecast comparison. Raw queue receipt private.

318,969 opened EuropeanSquares indexed rows in twelve localities, released detector
tracks, obs8/pred12 with raw annotation stride12. Image pixels only. No verified
metric scale, seconds, human gold, independent confirmation, physical safety,
true 3D or foundation claim. Prior two-source to four-source floor-producer shift
remains. Simulation not used. Stage5C/SMC remain off. Deployment unchanged.
