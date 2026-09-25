# Support Factorization: Historical Motion or Model-Generated Disagreement?

## Material Passport

Research execution, opened source-development data; not independent confirmation.
Registered before new decisions and their outcome readout, 25 September 2026.
Prior completed evidence: `european_causal_abstention_v1`, summary SHA256 bound in
the configuration. That study repaired a stopping defect but rejected the overall
utility claim for the support filter. Its negative results remain unchanged.

## Question and Fixed Comparisons

Which constraint removes useful neural interventions: observed-history support,
model-generated disagreement support, or requiring both to be supported by the
same fitting sources? This factorization tests the direct decision mechanism.
It does not, by itself, establish that two-source to four-source producer transport
causes the model-generated disagreement distribution shift.

Keep the stopped-state protection, all forecasters, utility/risk heads, fitting
population, score banks, floor, risk budget, source folds, sampling and evaluation
roles unchanged. Reuse the exact fitting-only source/state boxes: 1st/99th
percentiles, minimum 32 known fitting rows per box, at least two sources. No
threshold fitting or search, no producer retraining, no new neural model.

Feature axes are the preceding three causal scalars:
1. latest observed speed / mean observed speed;
2. mean observed turn angle;
3. mean neural-to-floor rollout disagreement / (12 * mean observed speed).

Four filters applied to the frozen stop-protected policy:
- **history**: support on axes 1 and 2 in at least two fitting sources;
- **disagreement**: support on axis 3 in at least two fitting sources;
- **separate**: both requirements above, possibly from different pairs of sources;
- **joint**: support on all axes in the same two or more sources, exactly the old combined rule.

Each has fixed-risk-rank and fixed-random controls with the same number of
interventions in each recording/current frame. Controls choose only among the
stop-protected policy's interventions. Preserve current cohort membership, not
future timestamps or whole-locality quotas. Include the unchanged stop policy.
Thirteen policies per parent; CV-target and both-floor-target parents; both
normalizers; three folds; seeds 17/29/43; all/easy events: **936 views**.
Of these, 288 reuse stop/joint/joint-risk/joint-random anchors and 648 are new
factor/control views. Replay all anchors exactly rather than renaming old runs.

## Readout

Freeze both full decision banks before either readout. Verify projected-source
memberships, decisions and current-frame count matching with separate scalar
arithmetic. Read future coordinates only for labels after that barrier.

Report all/easy/hard/complete ADE against protected floor and unchanged stop
policy, endpoint FDE against floor, positive-CV easy degradation, zero-CV harm,
unknown-label intervention counts, switch rate, per-locality p95/p99 and worst
locality. Compare each guard with count-matched controls; compare history,
disagreement and separate with joint. Three thousand paired locality-bootstrap
draws, seed 39271, existing eight-locality roster per fold. The 36 views per policy
are dependent; do not pool overlapping windows as independent samples or use
the best point/interval to select deployment.

For the joint rule, partition the stop-policy population into retained;
history-only failure; disagreement-only failure; both marginal failures; and
insufficient overlap of supporting sources. Outside-stop rows are separate.
For each category report known/unknown counts, foregone benefit and avoided
harm under a common per-locality floor-error denominator. Require the four
rejection categories to reproduce the complete old removal ledger exactly.
This attribution is a deterministic ablation on opened development data, not
causal identification of the training-process domain shift.

Evidence for a useful filter requires more than recovering accuracy relative to
an already harmful filter. Retain comparison with the unchanged stop policy,
equal-count controls, complete labels and hard/easy/tail costs. All results are
reported. No new deployment-selection criterion or independent safety guarantee.

## Execution and Remaining Gaps

Local native arm64 Python, four compute threads, one inter-op thread, zero loader
workers. Prior full comparison took about eighteen minutes of local evaluation
with ample memory; this fixed-forecast comparison is appropriate locally.
Per-group immutable artifacts, PID heartbeat, lock, resume, and 10-GiB free-disk
guard. Raw data, score banks, decisions and checkpoints remain private.

CREATE queue was freshly checked read-only. The simulation project confirmed that
an approved M3W remote directory remains unknown; no directory scan, submission,
authentication change or use of simulation resources is authorized by that handoff.
This limits remote inventory, not local progress. No remote absence claim.

Data remain 318,969 overlapping EuropeanSquares indexed rows in twelve opened
localities, released detector tracks, image pixels, obs8/pred12 rawstride12.
No metric/seconds, human-gold, true-3D, foundation or physical-safety claim.
Independent selection, calibration and confirmation remain closed. No historical
Stage37/SDD raw-t50 recertification. Stage5C and SMC remain off.
