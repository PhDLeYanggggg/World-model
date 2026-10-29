# Record-Balanced Constraint Readout

## Scope

Fresh_run: 18 neural policy fits, 36,000 updates, 732.452s, peak RSS 2,821,308,416B.
Cached_verified: fixed forecasts, causal feature packets, moments and frozen cost-sensitive/origin controls.
Not_run: new inner-held or independent evaluation, deployment, or new trajectory forecaster.
All 72 fitting views, 74 unique source-qualified recordings and three head seeds remain.
The 444 record/context/head views are repeated observations, not 444 independent recordings.
No recording is dropped. The locality082/51 fixed-candidate support obstruction remains.
Utility and2%budget definitions are unchanged. Equal recording weights alter training penalties only.

## Full-Policy Budgets

|Policy|Aggregate budgets|Record budgets|Known all/easy record violations|Undefined easy|Net easy degradation violations|
|---|---:|---:|---:|---:|---:|
|origin|1/72|16/444|331/164|73|329|
|cost_sensitive|54/72|84/444|346/54|14|0|
|record_balanced|52/72|131/444|289/39|9|0|

## Utility

Bounds cover unknown-label completions on fitting rows, not confidence intervals or FDE gain.
Masses are normalized per source/context/head view by full known reference cost, then macro-averaged.
Retention divides macro-averaged known benefit by the identically normalized eligible-positive upper bound.

|Policy|Known benefit pp|Known harm pp|Unknown envelope pp|Eligible positive benefit retained|
|---|---:|---:|---:|---:|
|origin|8.612003|0.677042|4.244517|43.800%|
|cost_sensitive|16.034646|0.547302|0.000926|81.552%|
|record_balanced|15.911456|0.558790|0.000693|80.925%|

|Contrast|Full lower/upper pp|Matched lower/upper pp|Empty matched|
|---|---:|---:|---:|
|record_balanced_minus_origin|+3.173285/+11.662125|-1.124007/+1.104923|0|
|record_balanced_minus_cost_sensitive|-0.135714/-0.133642|+0.020396/+0.021290|0|

Count matching assesses utility only. Empty matched risk remains not_estimable; it cannot pass safety.

## Recording Slices

|Slice|Origin|Cost-sensitive|Record-balanced|
|---|---:|---:|---:|
|source:eu-locality-007|0/18|16/18|17/18|
|source:eu-locality-008|8/168|12/168|30/168|
|source:eu-locality-020|1/6|6/6|6/6|
|source:eu-locality-048|0/6|6/6|6/6|
|source:eu-locality-067|3/18|12/18|13/18|
|source:eu-locality-074|4/192|2/192|29/192|
|source:eu-locality-082|0/6|0/6|0/6|
|source:eu-locality-110|0/6|6/6|6/6|
|source:eu-locality-112|0/6|6/6|6/6|
|source:eu-locality-119|0/6|6/6|6/6|
|source:eu-locality-124|0/6|6/6|6/6|
|source:eu-locality-126|0/6|6/6|6/6|
|seed:17|8/148|30/148|46/148|
|seed:29|0/148|27/148|40/148|
|seed:43|8/148|27/148|45/148|
|context:single0_seed43_controller1_dimensionless|2/99|18/99|35/99|
|context:single0_seed43_controller2_dimensionless|0/18|16/18|18/18|
|context:single1_seed43_controller0_dimensionless|10/99|18/99|20/99|
|context:single1_seed43_controller2_dimensionless|4/105|7/105|21/105|
|context:single2_seed43_controller0_dimensionless|0/18|18/18|17/18|
|context:single2_seed43_controller1_dimensionless|0/105|7/105|20/105|

## Verification

Independent arithmetic reconciles 1836 policy summaries. Readout 85.444s.
Controls reproduce their frozen whole-view and recording scores. No data role, threshold, or model choice follows from these results.
Engineering reproducibility is not independent generalization. No Stage5C, SMC, metric/seconds, true3D, foundation or submission-ready claim.
