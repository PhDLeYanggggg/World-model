# Post-Fit Constraint Diagnosis

This diagnosis uses frozen fitting scores and checkpoint metadata only. No new evaluation cohort is opened.
Counts repeat recording/forecast contexts across three head seeds, not independent recordings.

|Policy|Supported / required|Known all over|Known easy over|All upper over|Easy upper over|Undefined easy|Easy degradation over|
|---|---:|---:|---:|---:|---:|---:|---:|
|origin|16/444|331|164|408|330|73|329|
|exact_gradient|178/444|216|44|219|65|55|0|
|cost_sensitive|84/444|346|54|346|70|14|0|

## Source Concentration

|Source|Exact-gradient known all over / records|Cost-sensitive known all over / records|Cost-sensitive supported|
|---|---:|---:|---:|
|eu-locality-007|2/18|2/18|16|
|eu-locality-008|71/168|154/168|12|
|eu-locality-020|0/6|0/6|6|
|eu-locality-048|0/6|0/6|6|
|eu-locality-067|3/18|1/18|12|
|eu-locality-074|136/192|189/192|2|
|eu-locality-082|0/6|0/6|0|
|eu-locality-110|1/6|0/6|6|
|eu-locality-112|0/6|0/6|6|
|eu-locality-119|1/6|0/6|6|
|eu-locality-124|2/6|0/6|6|
|eu-locality-126|0/6|0/6|6|

## Dual Snapshot

Risk is measured just before the final optimizer update; duals are from just after it.
This is a training-mechanism diagnostic, not the final deployed-action safety score above.

|Policy/channel|Positive excess|Dual at cap|Both|Maximum dual|Median positive scaled excess|
|---|---:|---:|---:|---:|---:|
|exact_gradient/all|219|0|0|22.849|0.0008265808882665681|
|exact_gradient/easy|65|20|0|100|0.001438531873328082|
|cost_sensitive/all|344|0|0|25.8845|0.004253155510591584|
|cost_sensitive/easy|77|5|1|100|0.001736974168206646|

The budget remains 2%. Empty ratios remain not_estimable. Utility gains cannot waive the full-roster safety requirement.
This post-fit analysis motivates a next experiment; it does not choose a new model, threshold, or deployment policy.
