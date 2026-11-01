# Annotated Pair-Distance Accuracy

Secondary post-hoc diagnostic on frozen actions. Future positions/masks are evaluation labels only;no model,threshold or original gate changes.

## Full Retained Support

72 original groups;115,704,792 requested pair-step views,
87,548,430 observed and28,156,362 unknown.
These counts repeat contexts,critic seeds and overlapping windows. They are not independent pairs or additional training examples.
Zero-pair sources remain not_estimable and are not removed from the original ADE/risk population.

## Observed Labels

|Frozen policy|Mean normalized absolute pair-distance error on observed pair-steps|
|---|---:|
|floor|1.090051778|
|unary_joint|0.983701227|
|pair_joint|1.067189361|

Pair-aware versus unary observed error change:+8.487143%. This is a pooled point estimate,not a confidence interval.
Normalization uses the geometric mean of frozen past scales. Values are not meters,seconds or a physical collision rate.

## Missing Labels

Known contribution to mean gain over all requested pair-steps:-0.063171584.
Deterministic completion interval:[-0.195464276,+0.069121107].
The interval bounds missing outcomes using predicted-distance disagreement(reverse triangle inequality). It is not a bootstrap interval.
It crosses zero,so the unobserved full pair-distance outcome is not identified. Missing labels are not zero errors.

|Source|Known pair-step views|Unknown pair-step views|Known gain contribution|Completion lower|Completion upper|
|---|---:|---:|---:|---:|---:|
|eu-locality-007|23226|16878|-0.008618911|-0.035437961|+0.018200139|
|eu-locality-008|42223182|15293946|-0.053925249|-0.193859676|+0.086009178|
|eu-locality-020|10092|6684|-0.001460960|-0.012179447|+0.009257527|
|eu-locality-048|183096|117000|-0.064768795|-0.380462489|+0.250924899|
|eu-locality-067|618990|254946|-0.049465769|-0.255229997|+0.156298459|
|eu-locality-074|43815240|12193416|-0.072758380|-0.195614780|+0.050098021|
|eu-locality-082|0|0|not_estimable|not_estimable|not_estimable|
|eu-locality-110|119970|34038|-0.155902129|-0.303353778|-0.008450479|
|eu-locality-112|19620|13284|-0.016379120|-0.084735875|+0.051977634|
|eu-locality-119|279462|123810|-0.048121416|-0.134800584|+0.038557751|
|eu-locality-124|600|1416|+0.000162385|-0.014048905|+0.014373676|
|eu-locality-126|254952|100944|-0.071759020|-0.188337184|+0.044819143|

## Interpretation

The fixed contraction penalty reduces its own predicted geometric proxy,but observed pair-distance error and the main ADE utility both worsen.
This rejects the proxy as a sufficient repair in this experiment. It does not prove every interaction model fails or resolve missing outcomes.
The original all/easy risk remains failed. These are detector-silver image-coordinate SOURCE fits,not independent confirmation,human gold or physical safety.
All72 scalar groups are byte-verified in metric_groups.zip(135,962bytes);no trajectories,images,weights or future coordinates are included.
