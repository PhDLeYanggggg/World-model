# Current-query training: complete matched policy readout

## Material Passport

fresh_run:36 current-query Torch fits,72,000 updates plus4,000 exact refits,full72-group inference/allocation/scoring.
cached_verified:36 fixed-original-query controls,unchanged features,normalizers,draws,forecasts and original context.
Old control predictions are freshly re-inferred and hash-checked;their immutable choices are rescored,not reoptimized.
Readout PID73583: 628.182s;full replay PID75450: 623.356s.
All72 groups,144 fixed full-policy controls and the canonical summary reproduce exactly.
This is exposed SOURCE fitting/resubstitution,not independent evidence or new trajectory dynamics.

## Prespecified query-target effect

Conditional95% paired locality bootstrap,3,000 draws over12 exposed localities after context/head-seed averaging.
Primary:selected-set current-query joint minus fixed-original-query joint full completion-lower utility.

|Arm,current minus fixed query supervision|Full known increment pp [95% CI]|Full wholly-unknown completion-lower pp [95% CI]|
|---|---:|---:|
|row|-2.547757 [-4.424947, -0.674242]|-9.756862 [-18.757374, -4.240063]|
|set|-1.432011 [-3.335655, +0.608111]|-7.087372 [-13.238339, -2.904036]|

## Complete policy utility and unchanged risk

|Policy|Known ADE gain over floor [95% CI]|Supported recording-risk views|Known easy violations|Undefined easy|Selected views|
|---|---:|---:|---:|---:|---:|
|row_independent|+5.273807 [+3.371574, +7.507914]|13/444|260|23|98489|
|set_independent|+5.729492 [+3.498923, +8.500746]|12/444|282|21|137815|
|row_joint|+6.015268 [+3.803059, +8.621003]|14/444|286|23|98489|
|set_joint|+6.783336 [+4.179136, +9.978127]|11/444|313|21|137815|
|row_matched|+4.991092 [+3.022860, +7.445370]|23/444|266|33|75854|
|set_matched|+5.170120 [+2.971106, +7.987095]|24/444|267|33|75854|
|parent_independent|+1.904562 [+0.660504, +3.710170]|46/444|308|42|265671|
|parent_joint|+2.552950 [+0.861795, +4.959240]|43/444|324|42|265671|
|floor|+0.000000 [+0.000000, +0.000000]|0/444|0|444|0|
|fixed_query_row_joint|+8.563025 [+5.273775, +12.420742]|2/444|396|8|429205|
|fixed_query_set_joint|+8.215347 [+4.831265, +12.277071]|1/444|383|13|410575|

All original2% recording all/easy positive-harm/reference and easy-degradation requirements remain.
Common-count and generic trimmed comparisons are utility-only,not safety repairs. Empty comparisons are not estimable.
The moving loss can change its selected population;lower masked training loss alone does not establish calibration.
[Actual original-set probe,point loss,moving term and coverage](training_report.md).
Nonoptimal query views: {'row_joint': 5, 'row_matched': 3, 'set_joint': 2, 'set_matched': 2}.

## Context and cost-fit diagnostics

|Current-target contrast|Full known pp [95% CI]|Full completion-lower pp [95% CI]|
|---|---:|---:|
|set_joint_minus_row_joint|+0.768069 [+0.267183, +1.340767]|-3.525013 [-7.281836, -0.772340]|
|set_matched_minus_row_matched|+0.179028 [-0.159545, +0.584688]|-1.589234 [-2.936624, -0.477403]|

|Estimator|All-known-legal risk MSE [95% CI]|
|---|---:|
|original|+0.151659 [+0.097190, +0.219562]|
|row_local|+0.144729 [+0.099118, +0.205802]|
|selected_set|+0.143632 [+0.100179, +0.204305]|

This fitting MSE is neither probability ECE nor calibrated final-policy risk.
The machine-readable summary also retains FDE availability,all subsets and locality/seed contrasts;
per-record aggregates retain tails,unknown completion and predicted-path proxies. They are not relabelled
as complete-label,physical or independent evidence.

## Fitting checks

- complete36fresh_fits72groups: pass
- all144_fixed_query_controls_exact: pass
- paired_training: pass
- same_count_context_controls: pass
- current_minus_fixed_full_lower_CI_positive: fail
- positive_floor_lower_utility: fail
- complete_recording_risk: fail
- main_optima_certified: fail

4/8 fitting checks;advancement False.

## Interpretation limits

Moving-policy score regression uses stopped discrete choices;it is not an exact policy gradient,SPO control or calibration theorem.
Known-label ADE is not full-roster FDE;missing required endpoints remain not estimable.
Wholly-unknown completion bounds do not validate missing interior points. No unknown row is dropped or imputed.
The same149,510 target rows,18,766 queries,74 recordings and12 localities are reused. Three critic seeds share
one forecaster seed43. Overlapping windows/context views are not independent observations;intervals are nominal development summaries.
All seven forecasts remain fixed;no new physical-bank reoptimization with these heads or independent/SDD transfer is claimed.
Predicted-path geometry is a proxy,not physical safety or learned pair dynamics.
Task:obs8/pred12/rawstride12,European detector-silver/local-image coordinates,not historical raw-frame t+50.
No metric,seconds,human-gold,true3D,foundation,independent-success,deployment or submission-ready claim.
Independent roles,Stage5C execution and SMC stay closed.
