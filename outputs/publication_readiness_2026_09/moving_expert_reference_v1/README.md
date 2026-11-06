# Shared Frozen-Expert Context For Reference Learning

This is a completed model-side input contrast on already-exposed own-B.
All 48 reference fits and 48 complete exact refits precede policy readout.
Harm and label-support predictions remain bitwise frozen for every candidate.
Both new arms refit the same reference network; only shared expert context differs.

## Registered Primary

Expert-context minus history-refit joint-free paired completion-lower gain: **+0.133479 pp**,
exploratory 95% source-component interval **[-0.140981, +0.556238]**.
Eight source components, with three risk seeds averaged within source before
3,000 resamples. Teacher seed remains17; no independent C/H readout.

## Complete Policies

| Reference model | Decision | Known gain vs CV % | Separate lower % | Original risk supported | Known easy violations | Easy NA |
|---|---|---:|---:|---:|---:|---:|
| frozen | frame_uniform | +0.004273 | +0.003348 | 1/72 | 2 | 69 |
| frozen | independent | +0.225466 | +0.181711 | 28/72 | 13 | 7 |
| frozen | joint_free | +0.667496 | +0.587271 | 23/72 | 27 | 7 |
| frozen | joint_matched | +0.616239 | +0.548201 | 22/72 | 25 | 7 |
| frozen | proposal_only | +0.019827 | +0.019437 | 11/72 | 17 | 43 |
| history_refit | frame_uniform | +0.005670 | +0.004323 | 3/72 | 0 | 68 |
| history_refit | independent | +0.197905 | +0.160404 | 29/72 | 16 | 6 |
| history_refit | joint_free | +0.614917 | +0.541219 | 25/72 | 23 | 6 |
| history_refit | joint_matched | +0.554200 | +0.490429 | 20/72 | 28 | 8 |
| history_refit | proposal_only | +0.014617 | +0.014293 | 10/72 | 17 | 43 |
| expert_context | frame_uniform | +0.121836 | +0.119297 | 1/72 | 4 | 67 |
| expert_context | independent | +0.264958 | +0.214063 | 28/72 | 19 | 8 |
| expert_context | joint_free | +0.790637 | +0.711599 | 24/72 | 26 | 6 |
| expert_context | joint_matched | +0.732082 | +0.660103 | 18/72 | 33 | 8 |
| expert_context | proposal_only | +0.011645 | +0.009931 | 14/72 | 17 | 41 |

72 views repeat24recordings over3risk seeds; they are not72independent scenes.
Separate lower gains are not the paired lower contrast. Available-label
image-coordinate obs8/pred12ADE is not official complete12ADE or SDD raw t50.

Context joint-free worst defined signed easy-net degradation is 9.0168%,
and worst defined known-easy gross risk is 40.2986%; 6 easy ratios are NA.
These are distinct from mean gain and do not pass the original2%contract.

## Reference Prediction On The Same Known Candidate Cohort

| Model | R MSE | ER MSE | R bias | ER bias |
|---|---:|---:|---:|---:|
| frozen | +3.418436 | +0.214651 | -0.140591 | -0.004474 |
| history_refit | +3.360713 | +0.230431 | -0.167548 | -0.004916 |
| expert_context | +3.154554 | +0.280262 | -0.151116 | +0.006367 |

All entries use the same source-specific frozen known-only RMS and weights.
This is conditional reference prediction, not a calibrated risk bound.

## Selected-Risk Diagnosis

| Model | Decision | Pass | Observed harm violation | Completion-only | Missing support | ER overpredicted |
|---|---|---:|---:|---:|---:|---:|
| frozen | independent | 28 | 13 | 25 | 6 | 62 |
| frozen | joint_free | 23 | 27 | 16 | 6 | 61 |
| history_refit | independent | 29 | 16 | 21 | 6 | 63 |
| history_refit | joint_free | 25 | 23 | 19 | 5 | 60 |
| expert_context | independent | 28 | 19 | 17 | 8 | 65 |
| expert_context | joint_free | 24 | 26 | 18 | 4 | 62 |

Categories are mutually exclusive in observed-violation,completion-only,
support order. Undefined easy ratios remain NA, never zero harm. The linked
diagnostics also use identical frozen joint choices to separate reference
prediction changes from changed selection. Benefit caps can change utility;
there is no claim of a pure denominator-only effect.

## Verdict

Registered utility evidence: not supported.
Complete development candidate: **FAIL / NA**.
Original per-recording2%positive all/easy harm,signed easy-net and reference
support requirements are unchanged. Empty utility controls remain NA. No
identity-fallback exception, new test threshold or automatic deployment.

fresh_run:shared-context derivation,48fits,48full refits,new inference and readout.
cached_verified:parent features/labels,forecasts,role checkpoints and full controls.
not_run:new independent C/H,three forecaster seeds,new dynamics or deployment.
Original24-source-group main remains NA. Annotation-generation causality
unverified; no metric/seconds,true3D,foundation,human-gold,Stage5C or SMC claim.

[Protocol](protocol.md) | [Reproduction](operation_zh.md) | [Failure analysis](failure_analysis.md) |
[Model card](model_card.md) | [Data card](data_card.md) | [Verification](verification_report.md) | [Paper addendum](paper_addendum.md)
