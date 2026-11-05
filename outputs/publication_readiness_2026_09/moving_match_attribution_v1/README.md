# Where History Matches Differ

This descriptive follow-up reuses the frozen reference IDs and verified causal
features from moving_track_analogue_v1. No new retrieval, model selection or
held readout is performed. Known outcome labels stratify the analysis only;
they do not construct inference features or alter the original risk rules.

The120features are partitioned exactly into ego_history42, neighbor_history8,
rollout_summary14, rollout_history32 and candidate_neighbor24. Distances use
the original donor-only standardized/clipped features and reference weights.
All sources and empty strata are retained in analysis.json.

Source076 has722harmful and1,013jointly-nonpositive admitted known proposals.
The rollout/history block accounts for24.8% of squared matching distance in
the former versus16.0% in the latter. High-difference columns include candidate
versus floor lateral departure at steps3/6/12 and candidate end-side motion.
These are forecast features available before outcomes, not future truth.

This is not universal: source048 harmful and nonpositive admissions have very
similar block-distance shares; source059 harmful matches are dominated by ego
history; source060 highlights neighbor support and closing motion. Source065
has no admissions, so its corresponding strata remain NA. The analysis does
not prove that any named block causally determines the failure. Block dimension
also affects total distance share; per-feature squared contributions are kept.

The next controlled model check adjusts donor risk for query/donor feature
differences with a trained source-only regressor. It does not change references
or tune distance thresholds. The plain fitted regressor and a fixed ridge
transport control distinguish nonlinear prediction from transport itself.
Success is unproven; the neural transport protocol is registered separately.

Execution: PID47777,2.795s,peak RSS4,949,147,648bytes; observed exit0.
Two attribution tests pass. The separate neural preflight has three more tests,
including an actual optimizer/resume equivalence check; no real neural fitting
result is claimed by this note. The cache scan is local, not a new CREATE job.

Original2% risk/support and empty-control NA remain; original24primary is NA.
Only exposed own-B data; no independent confirmation, identity exemption,
deployment change, metric/seconds/true3D/foundation claim, Stage5C or SMC.
