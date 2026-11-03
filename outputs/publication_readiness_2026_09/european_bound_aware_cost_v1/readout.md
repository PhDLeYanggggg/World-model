# Matched Bound-Aware SOURCE Cost Learning

## Material Passport

Fresh_run36 full real Torch cost-head fits, both first-job full refits,
three-group recovery pilot, complete held cost/amplitude inference and
separate replay. Cached_verified EqMotion forecasts and SOURCE input roles.
This report aggregates public scalars, not fresh training/inference.

## Registered Result

Primary direct paired completion-lower effect: **-1.7031 [-5.7081, +1.1166] pp**.
Known-label paired effect: +0.3774 [-1.4814, +1.9475] pp.
The primary interval crosses zero; the registered advancement criterion
does not pass. Its negative point is not proof of a statistically significant
observed-ADE regression. The known-label difference is also inconclusive.

|Policy|Known gain vs fixed floor % [95% CI]|Individual completion-lower % [95% CI]|Supported/444|Worst signed easy-net degradation %|
|---|---:|---:|---:|---:|
|floor|+0.0000 [+0.0000, +0.0000]|+0.0000 [+0.0000, +0.0000]|0/444|0.0000|
|full_neural|+15.2748 [+11.1771, +19.5995]|+2.8440 [-12.3376, +13.2601]|0/444|235.8240|
|known_only_bounded|+3.0531 [+1.9643, +4.2948]|+1.0579 [-0.6163, +2.3406]|28/444|71.0389|
|bound_aware_bounded|+3.4304 [+2.3220, +4.6454]|+2.7669 [+1.6691, +3.9250]|108/444|43.6929|
|known_only_binary|+0.8341 [+0.4787, +1.2770]|+0.3936 [+0.0828, +0.7242]|13/444|12.1608|
|bound_aware_binary|+2.1216 [+1.1933, +3.1767]|+1.7678 [+0.9119, +2.7748]|25/444|44.0115|

Do not subtract the two individual lower bounds to replace the direct paired
primary. Unknown-row differences use disagreement between the two actual
policy paths, not subtraction of their separate worst-case bounds.
The zero-action floor adds no error but has no selected reference support;
its0/444 is not evidence that retaining the baseline is operationally unsafe.

## Risk And Intervention Trade-Off

|Bounded policy|Known all/easy violations|Completion all/easy violations|Undefined easy support|Active rate %|Mean amplitude, equal locality|
|---|---:|---:|---:|---:|---:|
|known_only_bounded|79/330|177/413|3|61.6463|0.213350|
|bound_aware_bounded|143/235|174/332|4|57.9310|0.239083|

Full-risk support improves28->108views, but known-all violations increase
79->143 while known-easy violations decrease330->235. This is a mixed trade-off,
not uniform safety improvement. No candidate satisfies the original contract.
There is no epsilon intervention, confidence search or selected favorable seed.

## Same-Target Fitting Check

|Fitted arm|Score against known-only target|Score against full bound target|
|---|---:|---:|
|known_only|0.901362|4.883677|
|bound_aware|1.011491|1.489479|

These are complete decoded fitting objectives using actual query-visit
multiplicities, then averaging eighteen jobs. Compare models within each
target column; unlike training objectives are not directly a performance
ranking. Both arms share preprocessing, initial weights, architecture,
query draws and full coverage budgets. Training/fitting errors are not
held-domain or selected-policy calibration results.

## Observed Failure Slices

|Locality|Known paired effect pp|Direct paired lower pp|
|---|---:|---:|
|eu-locality-007|+2.6776|+1.2116|
|eu-locality-008|+3.3189|+2.7520|
|eu-locality-020|+2.4223|+2.0944|
|eu-locality-048|+1.1052|+0.5131|
|eu-locality-067|+0.7628|+0.3489|
|eu-locality-074|-2.3278|-3.0381|
|eu-locality-082|-7.0221|-21.1166|
|eu-locality-110|+3.2659|+3.1684|
|eu-locality-112|-0.8552|-2.9266|
|eu-locality-119|+2.6991|+1.9031|
|eu-locality-124|+1.7654|-0.2205|
|eu-locality-126|-3.2837|-5.1262|

Worst signed easy-net view: eu-locality-008, recording77, single1_seed29_critic0.
Predicted easy-risk ratio is 0.059779% but known easy-risk is 57.7637%
and completion-easy upper is 96.6815%.
The predicted numerator and denominator are both estimates; this contrast
does not identify which one alone causes the error. It directly shows that
the nominal predicted guard is not calibrated to actual selected risk.

## Data Roles And Compute

For each forecaster groupF, critic fitting uses another groupG and evaluation
uses the remaining groupH. Both G choices rotate. Evaluation sites are in
neither forecaster nor critic fitting. All149,510SOURCE whitelist windows,
74recordings,twelve localities and actual seeds17/29/43 remain. The444
recording/pipeline views are repeats, not444independent scenes. SOURCE has
already informed research choices; this is not independent final confirmation.
Intervals use3000locality resamples after excluded-fold/seed averaging.
Gains use fitting-selected floor denominators,not the earlier CV denominator.

Primary updates:116160;full-refit updates:4000;pilot:1200.
Primary fitting PID9710:378.708s,peak9,240,887,296bytes.
Separate complete verification PID11539:124.657s including76.350s cost/amplitude inference.
All36cost/amplitude arrays and72metric groups replay exactly. Nativearm64
CPU4/inter-op1/workers0,atomic checkpoints,heartbeat and explicit resume.
These are cost heads,not new trajectory/JEPA/Transformer training.

## Verdict

Explicit bound targets materially improve their fitting objective and some
risk views,but do not repair complete policy safety or support the registered
paired primary. Keep deployment unchanged. Do not choose a binary arm or
favorable locality to rescue the failed primary. New reference-normalization,
threshold,amplitude or cohort rules require their own prospective protocol.
No calibration private arrays,final confirmation or DroneCrowd are accessed.
Masked ADE is not full12ADE; bounds cover wholly unknown rows,not missing
partial-future steps. Pixels/rawstride12/released silver only,not metric,
seconds,human gold,physical safety,true3D or foundation. Stage5C/SMC off.
