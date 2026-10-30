# Query-unit repair: complete paired policy readout

## Material Passport

fresh_run:36 real Torch critics,72,000 primary updates,4,000 exact-refit updates,complete72-group inference/allocation/scoring.
cached_verified:36 old critics and their immutable choices;old predictions freshly re-inferred and hash-checked,
and all144 full old-policy scores exactly recomputed. Old choices are NOT called freshly reoptimized.
Readout PID58848: 1746.472s;separate replay PID62325: 1788.801s.
Both complete runs agree on all72 groups and the canonical summary. No outcome-selected threshold or checkpoint.

## Primary unit-repair comparison

Conditional95% paired locality bootstrap,3,000 draws over12 exposed SOURCE localities.
Context/critic-seed views are averaged first;three critic seeds are not three forecaster producers.

|Arm, new units minus original units|Full known ADE increment pp [95% CI]|Full wholly-unknown completion-lower pp [95% CI]|
|---|---:|---:|
|row|+1.377776 [+0.875460, +1.896782]|-5.133639 [-14.047315, -0.163761]|
|set|+1.640505 [+1.101933, +2.251535]|-2.597889 [-7.642133, +0.293356]|

Selected-set full completion-lower increment is prespecified primary;row-local is a control.
No uncomputed difference-of-differences interval or special context/normalization interaction claim.

## Full policy utility and original risk

|Policy|Known ADE gain over floor [95% CI]|Supported recording-risk views|Known easy violations|Undefined easy|Selected views|
|---|---:|---:|---:|---:|---:|
|row_independent|+7.203862 [+4.427034, +10.743643]|2/444|362|8|429205|
|set_independent|+6.866192 [+4.048840, +10.528377]|1/444|349|13|410575|
|row_joint|+8.563025 [+5.273775, +12.420742]|2/444|396|8|429205|
|set_joint|+8.215347 [+4.831265, +12.277071]|1/444|383|13|410575|
|row_matched|+7.965236 [+4.732556, +11.785328]|3/444|384|12|377092|
|set_matched|+7.875175 [+4.546165, +11.860361]|2/444|378|13|377092|
|parent_independent|+1.904562 [+0.660504, +3.710170]|46/444|308|42|265671|
|parent_joint|+2.552950 [+0.861795, +4.959240]|43/444|324|42|265671|
|floor|+0.000000 [+0.000000, +0.000000]|0/444|0|444|0|
|old_units_row_joint|+7.185249 [+3.994640, +11.199603]|2/444|401|9|440480|
|old_units_set_joint|+6.574842 [+3.451302, +10.598514]|0/444|393|14|432796|

The original2% full-record all/easy harm/reference and easy-degradation criteria are unchanged.
Matched-count comparisons test utility only;empty comparisons are not estimable,not safe-zero outcomes.
[Actual original-unit training curves](training_report.md) do not substitute for final-selection calibration.

## Within-unit context controls

|New-unit contrast|Full known pp [95% CI]|Full completion-lower pp [95% CI]|
|---|---:|---:|---:|
|set_joint_minus_row_joint|-0.347678 [-0.617066, -0.028657]|-5.657865 [-13.058285, -1.481855]|
|set_matched_minus_row_matched|-0.090061 [-0.274255, +0.151813]|-3.636523 [-8.482949, -0.883135]|

Nonoptimal query views: {'row_joint': 7, 'row_matched': 7, 'set_joint': 3, 'set_matched': 5}.
Selected-set joint decision reasons: {'exact_milp': 60585, 'zero_count_exact': 41305, 'unconstrained_optimum_feasible': 7579, 'singleton_pointwise_exact': 3124, 'postcheck_failed_parent_fallback': 3}.

## Fitting gates

- complete36fresh_fits72groups: pass
- all144old_critic_controls_exact: pass
- original_matched_training: pass
- same_count_context_controls: pass
- new_units_set_minus_old_units_lower_CI_positive: fail
- positive_floor_lower_utility: fail
- complete_recording_risk: fail
- main_optima_certified: fail

4/8 fitting checks;advancement False.

## Scope and limits

Exposed SOURCE fitting/resubstitution only,not independent confirmation or whole-pipeline OOF.
All149,510 unique target rows,18,766 current-frame queries,74 recordings and12 localities remain.
Repeated views and overlapping windows are not independent observations. ADE uses available labels;
full-roster FDE is not estimable for missing endpoints. Wholly-unknown bounds do not certify missing interior labels.
Training uses original selected sets;final allocation can select a different set. Normalizing a loss correctly
does not guarantee final-selection calibration or finite-record risk. No risk threshold or decoder constraint is relaxed.
The same seven forecasts remain fixed;this experiment trains risk critics,not new trajectory dynamics.
Predicted-path geometry is a proxy,not verified physical safety or all-visible-agent modeling.
Task:obs8/pred12 at raw stride12,detector-silver/local-image coordinates,not historical raw-frame t+50.
No metric,seconds-level,human-gold,true3D,foundation,independent-success or submission-ready claim.
Deployment stays unchanged;independent roles,Stage5C execution and SMC remain closed.
