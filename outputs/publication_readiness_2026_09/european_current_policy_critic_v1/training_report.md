# Current-query supervision: real training telemetry

## Material Passport

fresh_run:36 native Torch risk fits,72,000 primary updates,4,000 separate full-refit updates.
cached_verified:36 fixed-query critic controls,unchanged source inputs,forecasts,initial states and draws.
Training PID69758: 1877.402s,peak RSS10798153728B.
Only query supervision changes to current decoded choices,detached from gradients. Context remains original.
Both full first-head2,000-step refits agree exactly. All unknown draws have zero supervised loss.

## Comparable fixed-original-set probe

These16 fixed fitting-monitor queries per head are not validation/test or full-dataset means.
The ORIGINAL selected set and original error units are common in this probe. New moving-set
loss values are not substituted for that probe or called the same-population error.

|Arm|Training target|Initial common probe|Final common probe|Heads decreased|Final point loss|Query/point gradient median|
|---|---|---:|---:|---:|---:|---:|
|row_local|fixed|0.000435536|0.000144110|15/18|0.078885021|1.7576483894424197|
|row_local|current|0.000435536|0.000600934|8/18|0.109814861|2.104063414505508|
|selected_set|fixed|0.000435536|0.000153624|16/18|0.078215335|1.6346242439618357|
|selected_set|current|0.000435536|0.000477941|10/18|0.113005718|2.106301681094304|

## Moving target and supervision coverage

|Arm|Final moving masked query term,original units|Initial selected rows|Final selected rows|Final complete queries(out of16)|
|---|---:|---:|---:|---:|
|row_local|0.000179249|19.5|12.5|6.5|
|selected_set|0.000327683|19.5|16.0|7.5|

The moving registered loss averages over the sampled queries and gives no supervised contribution
to empty or selected-unknown sets. Zero contribution is NOT zero risk. Changing loss can reflect
changing choices/support,not only better prediction. Fixed-control moving-set curves were not
logged and are not invented. Full final-policy utility and original recording-risk tests remain decisive.
Gradient ratios are actual same-batch component norms;the supervised populations differ by target.

## Scope

Exposed SOURCE fitting/resubstitution,not independent or whole-pipeline OOF confirmation.
Final step2,000 is fixed;no loss-based checkpoint or threshold selection.
No new trajectory forecaster,physical guarantee,metric/seconds,human-gold,true3D,foundation or deployment claim.
Task:obs8/pred12/rawstride12,detector-silver/local-image coordinates. Stage5C execution/SMC stay off.
