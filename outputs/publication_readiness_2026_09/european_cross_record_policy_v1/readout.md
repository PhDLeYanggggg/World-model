# Cross-Recording Meta-Policy Readout

## Evidence Scope

Fresh_run: 36 neural policy fits, 72,000 updates; training 3508.268s.
Peak RSS 3,687,137,280 bytes; readout 99.382s.
Cached_verified: frozen source data, original forecasters/moments, exact_gradient and origin controls.
Not_run: independent test, new trajectory forecaster, deployment. Head seeds are not forecasting seeds.
Both episode sides are TRAIN. No test-time adaptation. All finals frozen before the exposed inner-held readout.
All/easy positive-harm budgets and easy-degradation requirement remain 2%; matched utility is not safety.
Empty matched risk is not_estimable. Completion bounds are not CIs or FDE improvements.

## Full-Policy Safety

|Scope|Policy|Budget views|Record budget views|Known all/easy violations|Easy degradation violations|Undefined easy|Actions / unknown|
|---|---|---:|---:|---:|---:|---:|---:|
|fit|floor|0/72|0/444|0/0|0|72|0/0|
|fit|origin|1/72|16/444|35/25|68|7|62902/1567|
|fit|exact_gradient|53/72|178/444|13/0|0|6|138260/228|
|fit|same_record_meta|51/72|170/444|15/1|0|6|140846/263|
|fit|cross_record_meta|51/72|160/444|15/0|0|6|144025/268|
|inner_held|floor|0/72|0/186|0/0|0|72|0/0|
|inner_held|origin|0/72|3/186|61/44|61|14|32621/847|
|inner_held|exact_gradient|0/72|2/186|64/60|49|3|56720/869|
|inner_held|same_record_meta|0/72|2/186|63/59|55|3|59385/922|
|inner_held|cross_record_meta|0/72|4/186|63/59|50|3|57618/902|

## Utility Completion Bounds

Equal-weight view means; not pooled row accuracy.

|Scope|Contrast|Full lower / upper pp|Count-matched lower / upper pp|Empty matched views|
|---|---|---:|---:|---:|
|fit|cross_record_meta_minus_origin|-7.757760 / +0.731863|-2.574791 / -1.087950|3|
|fit|cross_record_meta_minus_exact_gradient|-0.496860 / -0.496240|-0.112021 / -0.111635|3|
|fit|cross_record_meta_minus_same_record_meta|-0.869973 / -0.869183|-0.107016 / -0.106565|3|
|fit|same_record_meta_minus_exact_gradient|+0.372743 / +0.373312|-0.000395 / -0.000022|3|
|inner_held|cross_record_meta_minus_origin|-4.386370 / +0.598148|-1.571132 / -0.745989|7|
|inner_held|cross_record_meta_minus_exact_gradient|-0.723173 / +0.310115|-0.159391 / -0.031468|3|
|inner_held|cross_record_meta_minus_same_record_meta|-1.062365 / +0.162885|-0.188814 / -0.044687|3|
|inner_held|same_record_meta_minus_exact_gradient|-0.314186 / +0.800608|-0.044339 / +0.093899|3|

## Paired Source Bootstrap

3,000 draws of 12 source clusters, seed 1342177. Average repeated contexts/head seeds within source first.
Intervals below combine source-bootstrap endpoint uncertainty with outcome-completion bounds; they are exploratory, not independent confirmation.
A positive upper endpoint is not a positive result. No row/bootstrap independence claim.

|Held contrast|Full outer envelope pp|Matched outer envelope pp|
|---|---:|---:|
|cross_record_meta_minus_origin|-7.106211 / +3.945713|-2.342835 / -0.232816|
|cross_record_meta_minus_exact_gradient|-1.295729 / +0.923075|-0.333954 / +0.036209|
|cross_record_meta_minus_same_record_meta|-1.875871 / +1.036292|-0.453565 / +0.063353|
|same_record_meta_minus_exact_gradient|-0.841236 / +1.469480|-0.109715 / +0.178024|

## Held Source and Seed Slices

|Slice|Primary budget views|Primary known all/easy violations|Cross minus same matched lower / upper pp|
|---|---:|---:|---:|
|source:eu-locality-007|0/6|6/6|+0.021672 / +0.026623|
|source:eu-locality-008|0/6|6/6|-0.078207 / +0.076036|
|source:eu-locality-020|0/6|3/3|+0.000000 / +0.000000|
|source:eu-locality-048|0/6|6/6|-0.108039 / +0.259377|
|source:eu-locality-067|0/6|4/6|-0.054730 / -0.054730|
|source:eu-locality-074|0/6|6/6|-0.046330 / +0.023900|
|source:eu-locality-082|0/6|3/0|+0.000000 / +0.000000|
|source:eu-locality-110|0/6|5/6|-1.365031 / -0.774958|
|source:eu-locality-112|0/6|6/6|-0.084892 / +0.032772|
|source:eu-locality-119|0/6|6/6|-0.569228 / -0.144275|
|source:eu-locality-124|0/6|6/5|+0.067982 / +0.067982|
|source:eu-locality-126|0/6|6/3|-0.048970 / -0.048970|
|seed:17|0/24|22/19|-0.284453 / -0.112193|
|seed:29|0/24|21/20|-0.118088 / +0.018244|
|seed:43|0/24|20/20|-0.163902 / -0.040111|

## Known Benefit and Harm Retained

Normalized masses, relative to frozen origin; unknown outcomes remain separately bounded.

|Scope|Policy|Benefit retained|Harm retained|Known net change pp|
|---|---|---:|---:|---:|
|fit|exact_gradient|59.356%|28.527%|-3.016398|
|fit|same_record_meta|63.949%|31.859%|-2.643371|
|fit|cross_record_meta|53.565%|28.214%|-3.512949|
|inner_held|exact_gradient|52.963%|52.603%|-1.687582|
|inner_held|same_record_meta|59.213%|57.595%|-1.444371|
|inner_held|cross_record_meta|47.151%|46.610%|-1.894111|

## Compute and Checks

|Arm|Fits|Updates|Seconds|Support+query forward row occurrences|Final clipped inner sides|
|---|---:|---:|---:|---:|---:|
|same_record_meta|18|36000|1744.709|3579900000|0|
|cross_record_meta|18|36000|1683.886|3579900000|0|

Separate arithmetic checks: 5022 policy summaries, 288 frozen controls.
Primary full-budget support: False. Deployment remains unchanged.
74 fit / 31 held source-qualified recordings repeat across contexts and head seeds.
No independent calibration, confirmation, metric/seconds, true-3D, foundation or submission-readiness claim.
Stage5C not executed; SMC not enabled.
