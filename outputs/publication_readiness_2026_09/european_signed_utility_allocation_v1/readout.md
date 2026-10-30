# Joint utility admissibility on frozen risks

## Material Passport

fresh_run:72 complete causal allocation/scoring groups and a separate full replay.
cached_verified:18 fixed selected-set critics,all source inputs/forecasts,and original full-policy choices.
not_run:new training,trajectory dynamics,independent calibration/confirmation or deployment.
Readout PID80693:1564.756s,peakRSS11025170432B.
Replay PID84033:1563.212s,peakRSS11267358720B.
All72 groups,72 original full-policy controls,148 unique record/context oracle controls and the summary agree exactly.

## Controlled factor

Only per-action positive utility is removed from the signed admissible set. Positive reference/easy-reference,
at most one action per agent,nonnegative total predicted utility,and both ORIGINAL2% positive-harm/reference
inequalities remain. Benefit never cancels positive harm in either numerator. All predictions are frozen.
The matched pair retains exactly the same intervention count in every current query;free counts are secondary.

## Prespecified paired effects

|Signed minus positive-action control|Known utility pp [95% CI]|Wholly-unknown completion-lower pp [95% CI]|
|---|---:|---:|
|matched|+0.010488 [+0.002104,+0.019958]|-0.007179 [-0.024395,+0.007941]|
|free|+0.043942 [+0.025620,+0.062978]|+0.009636 [-0.022548,+0.035381]|

Primary:matched full completion-lower effect. Nominal3,000-draw paired locality bootstrap over12
EXPOSED SOURCE localities after context/head-seed averaging;not adaptive-search-adjusted confirmation.

|Policy|Known ADE gain over floor [95% CI]|FDE gain [95% CI]|Supported original risk views|Known easy violations|Undefined easy|Selected views|
|---|---:|---:|---:|---:|---:|---:|
|positive_matched|+8.215347 [+4.831265,+12.277071]|not_estimable|1/444|383|13|410575|
|signed_matched|+8.225835 [+4.838207,+12.289703]|not_estimable|1/444|383|13|410575|
|positive_free|+8.528155 [+5.041867,+12.650426]|not_estimable|1/444|385|12|520950|
|signed_free|+8.572097 [+5.066425,+12.694224]|not_estimable|1/444|385|11|535025|

## Label-side structural diagnostic

All148 unique record/context views are retained. Unrestricted known-label oracle:
148 supported,148 positive feasible. True-positive-action-only oracle:
148 supported,148 positive feasible.
The unrestricted control is freshly recomputed and verified against the existing result;it is not newly discovered
opportunity. The restricted oracle uses future costs only for diagnosis,not model inference or selection.
It differs from a conditional expected-moment policy. Neither feasible oracle proves causal learnability.

## Fitting checks

- complete72: pass
- all72_original_controls_exact: pass
- oracle_control_exact: pass
- matched_count_exact: pass
- primary_lower_CI_positive: fail
- floor_lower_CI_positive: fail
- complete_original_risk: fail
- primary_optima_certified: fail

Advancement:False. Deployment unchanged.

Oracle utility comparison,head seeds deduplicated:{'comparable_record_contexts': 148, 'exact_equal': 148, 'max_abs_gap': 0.0}

## Numerical and statistical limits

Solver reason counts:{'signed_matched': {'exact_milp': 63642, 'zero_count_or_no_action_exact': 41305, 'unconstrained_optimum_feasible': 7579, 'postcheck_failed_anchor_fallback': 70}, 'positive_free': {'exact_milp': 75695, 'unconstrained_optimum_feasible': 27485, 'singleton_pointwise_exact': 9410, 'postcheck_failed_parent_fallback': 6}, 'signed_free': {'exact_milp': 85083, 'unconstrained_optimum_feasible': 26381, 'zero_count_or_no_action_exact': 1104, 'postcheck_failed_anchor_fallback': 28}}
A fallback retains an actual feasible incumbent,not a global-optimality claim. Predicted nesting is not observed safety.
Empty matched risk is not_estimable;count controls do not replace original complete recording risk.
No independent rows are created by repeated contexts,seeds or overlapping windows. Three critic seeds share
forecast producer43. All149,510 target rows,74 recordings and12 localities remain;no failing source is dropped.
Unknown selected outcomes and missing endpoints remain unobserved. Bounds do not complete interior labels.
The summary retains supplementary hard/easy slices;they do not replace the original risk requirement.
This decoder-only experiment does not newly score interaction-path/tail proxies or SDD safety;those are not_run
here,not borrowed from older policies. No world-dynamics contribution or complete benchmark is claimed.
Obs8/pred12/rawstride12,oblique detector-silver/local-image coordinates,not SDDt50,metric/seconds,human gold,
true3D,foundation,independent success or submission-ready evidence. Independent roles,Stage5C and SMC stay closed.

[Full-policy figure](figures/allocation_result.svg),[mechanism counterexample](mechanism_note.md),
[registered protocol](protocol.md),[Chinese reproduction guide](reproduction_zh.md).
