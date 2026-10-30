# Selected-query risk critic: complete paired readout

## Material Passport

fresh_run:36 matched real Torch risk fits,72,000 primary updates,4,000 exact-refit updates,full inference/allocation/scoring.
cached_verified: frozen upstream forecasts,cost heads,causal features,original selected sets and source inputs.
This is exposed SOURCE fitting/resubstitution,not independent confirmation or a new trajectory dynamics model.
Readout PID43330: 1744.334s;separate replay PID51271: 1696.211s.
All72 groups and144 old full-policy controls match. No outcome-based checkpoint or threshold selection.

## Controlled method

The two30,726-parameter heads differ only in row-local versus original-selected-set hidden context.
Both receive count metadata and learn the same all-action/selected-query objective with the same fitting-only RMS.
All seven candidates are rescored. Benefit/reference/easy-reference are exactly frozen;harm/easy-harm change.
The same joint allocator uses each head's independent per-query count;extra matched controls use their common minimum.
Labels and eventual support masks do not enter inference/context. Unknown rows remain in the cohort and context.

## Full-policy utility and unchanged risk

Known ADE gain versus the frozen floor (%);conditional95% locality bootstrap,3,000 draws over12 exposed localities.
Repeated context/critic-seed views are averaged first. There is one fixed forecaster seed,not three producers.

|Policy|Known ADE gain [95% CI]|Completion-lower floor gain pp [95% CI]|Supported record views|Known easy violations|Undefined easy|Selected views|
|---|---:|---:|---:|---:|---:|---:|
|row_independent|+5.818984 [+3.185088, +9.269372]|-4.466850 [-21.015210, +5.942419]|5/444|395|9|440480|
|set_independent|+5.378410 [+2.799399, +8.838001]|-6.215542 [-26.232325, +5.877955]|2/444|384|14|432796|
|row_joint|+7.185249 [+3.994640, +11.199603]|-3.647100 [-20.517782, +7.106495]|2/444|401|9|440480|
|set_joint|+6.574842 [+3.451302, +10.598514]|-5.516051 [-25.892232, +6.956512]|0/444|393|14|432796|
|row_matched|+6.817770 [+3.601538, +10.835636]|-3.610179 [-20.037377, +6.935580]|2/444|391|14|399538|
|set_matched|+6.376949 [+3.276380, +10.378589]|-3.591708 [-19.826546, +6.803869]|1/444|387|15|399538|
|parent_independent|+1.904562 [+0.660504, +3.710170]|+1.263345 [-0.277283, +3.174181]|46/444|308|42|265671|
|parent_joint|+2.552950 [+0.861795, +4.959240]|+1.786960 [-0.151977, +4.306964]|43/444|324|42|265671|
|floor|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|0/444|0|444|0|

|Prespecified contrast|Full known increment pp [95% CI]|Full wholly-unknown completion-lower pp [95% CI]|
|---|---:|---:|
|set_joint_minus_row_joint|-0.610407 [-0.880023, -0.349336]|-4.872588 [-11.205362, -1.301837]|
|set_matched_minus_row_matched|-0.440822 [-0.714955, -0.193132]|-2.201911 [-3.941785, -0.879762]|
|set_joint_minus_parent_joint|+4.021891 [+2.505785, +5.691857]|-7.386733 [-26.103590, +3.105270]|
|row_joint_minus_parent_joint|+4.632298 [+3.002342, +6.366206]|-5.497561 [-20.719781, +3.389788]|
|set_joint_minus_set_independent|+1.196432 [+0.613542, +1.807569]|+0.502880 [-0.122353, +1.135591]|

The context-versus-local full-policy contrast is primary. The explicit matched control is utility-only.
Generic empty count-trimmed contrasts remain not estimable in the aggregate archive;no zero-risk certificate is invented.
A context-parent improvement alone cannot isolate pooling:both critics also share new supervision,normalization and architecture.

## Risk learning and actual training

|Cost estimator|All known legal candidate risk MSE [95% CI]|
|---|---:|
|original|+0.151659 [+0.097190, +0.219562]|
|row_local|+0.107182 [+0.064435, +0.167201]|
|selected_set|+0.108521 [+0.065931, +0.168496]|

MSE uses fitting-only causal scales and risk RMS. It is not ECE,a calibrated probability or observed safety.
[Actual training curves and gradient telemetry](training_report.md) are separate from policy efficacy.
Original recording failure conditions for selected-set joint: {'all_selected_risk_upper': 435, 'easy_selected_risk_upper': 433, 'easy_degradation_upper': 401}.
Nonoptimal query views: {'row_joint': 2, 'row_matched': 2, 'set_joint': 8, 'set_matched': 6}.
Feasible anchors are retained on solver rejection. Original all/easy2% and easy-degradation upper criteria remain.

## Fitting gates

- complete72groups: pass
- all144parent_full_policy_controls_exact: pass
- paired_feature_initialization_and_training_draws: pass
- same_count_controls_every_query: pass
- context_minus_row_full_lower_CI_positive: fail
- context_vs_floor_completion_lower_CI_positive: fail
- complete_context_recording_risk: fail
- all_main_query_optima_certified: fail

4/8 fitting checks;advancement False.
No deployment or independent-evaluation promotion follows merely from loss reduction or fewer interventions.

## Limits

The same149,510 unique agent-target rows and18,766 current-frame queries across74 recordings/12 localities
generate reused views. Full-roster FDE is not estimable if a required endpoint is unavailable.
ADE uses available future labels;wholly-unknown completion bounds do not certify missing interior labels.
The query target concerns the original selected set;the recalibrated decoder may select a different set.
Predicted budget feasibility does not prove calibration under that final selection or finite-recording compliance.
Pair geometry is a predicted-path proxy among target queries,not verified physical safety or all-visible-agent modeling.
New physical-only reoptimization with these critics is not run;the preceding bank-control result cannot be relabelled as such.
This is obs8/pred12 at raw stride12,not historical raw-frame t+50. No metric,seconds,human-gold,true3D or foundation claim.
Independent roles,Stage5C execution and SMC remain closed;submission readiness remains false.
