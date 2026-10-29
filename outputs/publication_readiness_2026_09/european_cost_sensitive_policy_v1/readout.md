# Cost-Sensitive Policy: Fitting Readout

## Scope

Fresh_run: 18 neural policy fits, 36,000 updates, 604.904s, peak RSS 3,015,131,136B.
Cached_verified: fixed forecasts, source data, moments and exact-gradient/origin controls.
Not_run: new inner-held evaluation, independent calibration/test, deployment or trajectory forecaster training.
All 72 fitting views and 74 source-qualified recordings remain. Three head seeds are not forecasting seeds.
The same locality082/51 action/support obstruction remains; no row or recording was dropped.
All/easy positive-harm and easy-degradation budgets remain 2%. Empty risk is not_estimable.
Utility completion bounds below are TRAIN diagnostics, not FDE improvements or confidence intervals.

## Exact Fitting Budgets

|Policy|View budgets|Record budgets|Known all/easy violations|Easy degradation violations|Actions / unknown|
|---|---:|---:|---:|---:|---:|
|origin|1/72|16/444|35/25|68|62902/1567|
|exact_gradient|53/72|178/444|13/0|0|138260/228|
|cost_sensitive|54/72|84/444|12/0|0|293781/395|

## Utility and Retention

|Contrast|Full lower/upper pp|Matched lower/upper pp|Empty matched|
|---|---:|---:|---:|
|cost_sensitive_minus_origin|+3.307900/+11.796866|-1.263524/+0.981568|0|
|cost_sensitive_minus_exact_gradient|+10.567627/+10.569935|+1.824042/+1.824777|3|

|Policy|Known benefit pp|Known harm pp|Unknown envelope pp|Eligible positive benefit retained|
|---|---:|---:|---:|---:|
|origin|8.612003|0.677042|4.244517|43.800%|
|exact_gradient|5.111704|0.193142|0.000457|25.998%|
|cost_sensitive|16.034646|0.547302|0.000926|81.552%|

## Slices

Each cell is passing/required fitting views. No source/seed selection.

|Slice|Origin|Exact gradient|Cost-sensitive|
|---|---:|---:|---:|
|source:eu-locality-007|0/6|6/6|6/6|
|source:eu-locality-008|0/6|3/6|0/6|
|source:eu-locality-020|1/6|6/6|6/6|
|source:eu-locality-048|0/6|6/6|6/6|
|source:eu-locality-067|0/6|6/6|6/6|
|source:eu-locality-074|0/6|0/6|0/6|
|source:eu-locality-082|0/6|0/6|0/6|
|source:eu-locality-110|0/6|5/6|6/6|
|source:eu-locality-112|0/6|6/6|6/6|
|source:eu-locality-119|0/6|5/6|6/6|
|source:eu-locality-124|0/6|4/6|6/6|
|source:eu-locality-126|0/6|6/6|6/6|
|seed:17|1/24|18/24|18/24|
|seed:29|0/24|17/24|18/24|
|seed:43|0/24|18/24|18/24|
|context:single0_seed43_controller1_dimensionless|0/12|12/12|9/12|
|context:single0_seed43_controller2_dimensionless|0/12|10/12|12/12|
|context:single1_seed43_controller0_dimensionless|1/12|9/12|9/12|
|context:single1_seed43_controller2_dimensionless|0/12|6/12|6/12|
|context:single2_seed43_controller0_dimensionless|0/12|10/12|12/12|
|context:single2_seed43_controller1_dimensionless|0/12|6/12|6/12|

## Mechanism Check

These are weighted absolute logit derivatives at each model's own final TRAIN duals, not parameter gradients or a causal attribution of failure.
Saturation threshold p(1-p)<1e-4 is diagnostic only, not an inference threshold.

|State|Wrong choices|Wrong saturated choices|Mean wrong cost|STE derivative|Cost-logistic derivative|
|---|---:|---:|---:|---:|---:|
|exact_gradient|508965|443891|0.369643|0.00121473|1.47691|
|cost_sensitive|405645|25925|0.0943869|0.0404206|0.31757|

## Verification and Claims

Separate arithmetic checks reproduce 1836 policy summaries. The readout takes 78.580s.
The oracle has been exhaustively checked on small cases; actual controls replay at complete-view and record level.
Even improved fitting would not repair prior held-data failure or establish scene-joint decision generalization.
No held-data CI is computed for an unrun held comparison. No deployment, Stage5C, SMC, metric/seconds, true3D, foundation or paper-readiness claim.
