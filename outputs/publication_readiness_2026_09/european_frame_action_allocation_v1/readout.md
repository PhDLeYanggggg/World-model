# Current-query multi-action allocation: complete fixed readout

## Material Passport

Fresh run:real causal inference, fixed allocation and label-aware diagnostic recomputation.
Cached verified:18 already trained unit-admission cost heads, their unchanged forecast bank and inputs.
New training updates:0. SOURCE fitting/resubstitution only; no independent or whole-pipeline OOF claim.

## Controlled question

Keep all predictors and costs fixed. Can pooling the all/easy predicted2% risk budget
within the same current recording/frame query improve on independent decisions?
The primary joint arm uses exactly the same intervention count in EVERY query.
Uniform uses one action for all query targets; free joint allocation may change count.
These are query targets, not guaranteed all visible agents. No pair-interaction model is trained.

Complete readout PID19894 takes729.535s at peak RSS10922934272B.
Separate full replay PID22148 takes741.428s.
All72 prediction/action/full-policy controls and all444 repeated record-oracle checks agree.
Exact replay validates computation, not independent scientific confirmation.

## Complete-policy utility and original risk

Known ADE gains (%) versus the frozen floor;95% bootstrap CI over12localities,
3,000 draws after context/seed averaging. Three head seeds are not independent recordings.

|Decoder|Known ADE gain|Supported recording views|Known easy violations|Undefined easy support|Selected row views|
|---|---:|---:|---:|---:|---:|
|independent|+1.904562 [+0.660504,+3.710170]|46/444|308|42|265671|
|frame_uniform|+0.626333 [+0.185203,+1.270975]|112/444|156|139|14088|
|joint_matched|+2.552950 [+0.861795,+4.959240]|43/444|324|42|265671|
|joint_free|+2.704414 [+0.962400,+5.167592]|39/444|322|42|327888|
|floor|+0.000000 [+0.000000,+0.000000]|0/444|0|444|0|

|Fixed contrast|Known increment pp [95% CI]|Wholly-unknown completion-lower pp [95% CI]|
|---|---:|---:|
|joint_matched_minus_independent|+0.648388 [+0.181208,+1.272930]|+0.506055 [+0.071211,+1.107688]|
|joint_free_minus_independent|+0.799852 [+0.266361,+1.491805]|+0.626490 [+0.128927,+1.296794]|
|uniform_minus_independent|-1.278229 [-2.457214,-0.354037]|-1.560936 [-2.766406,-0.606873]|
|joint_matched_minus_uniform|+1.926617 [+0.535650,+3.738000]|+1.548378 [+0.194746,+3.352707]|

The no-intervention floor has zero utility change; its selected risk denominator
is undefined, not a selected-risk safety certificate. Empty matched controls are not
estimable and count matching is utility-only. The original complete-policy recording
risk requirements are not replaced by predicted budgets or a positive average.

## Allocation mechanism and feasible opportunity

Repeated query views:112596;multi-agent:86316;
changed primary queries:36305;nonoptimal primary queries:8.
Solver/fast-path reasons:{'joint_matched': {'zero_count_exact': 50332, 'exact_milp': 53688, 'singleton_pointwise_exact': 2045, 'unconstrained_optimum_feasible': 6523, 'postcheck_failed_parent_fallback': 8}, 'joint_free': {'exact_milp': 69428, 'singleton_pointwise_exact': 6571, 'unconstrained_optimum_feasible': 36591, 'postcheck_failed_parent_fallback': 6}}.
Seed-deduplicated label-aware frame diagnostic:{'queries': 37532, 'algebraic_budget_failures': 0, 'defined_all_support': 35305, 'defined_easy_support': 20858, 'positive_gain_queries': 35305}.
The oracle is the same whole-record label-aware allocation, freshly reproduced then
checked frame by frame. Algebraic budget satisfaction with no easy support is not
a defined easy-risk pass. This is neither an inference policy nor a matched-rate optimum.
Primary overlapping recording failure conditions:{'all_selected_risk_upper': 338, 'easy_selected_risk_upper': 388, 'easy_degradation_upper': 271}.

## Limits and fitting gates

The same149,510unique queries,74recordings and12localities generate reused context/seed views.
ADE uses available future positions. Full-roster FDE is not estimable when a required
view lacks final labels. Completion sensitivity covers wholly unknown rows, not all
missing interior positions in partially observed ADE labels. No outcomes are imputed.
Pair-distance contraction in each aggregate group is a predicted-path proxy among
query targets, normalized by a frozen fitting scale, not verified physical safety.

- complete72groups:pass
- all_frozen_controls_exact:pass
- same_intervention_count_every_query:pass
- all_primary_queries_solved_optimally:fail
- primary_known_CI_positive:pass
- primary_completion_lower_CI_positive:pass
- complete_primary_recording_risk:fail

Fitting gates:5/7;advancement:False.
Independent confirmation, deployment change and submission readiness remain false.
Task:8observed/12requested steps at raw stride12, not supplemental rawt+50.
Detector-silver top-down pixel/raw-frame data are not human gold,metric,seconds-level,
true3D or foundation evidence. No Stage5C execution or SMC.
