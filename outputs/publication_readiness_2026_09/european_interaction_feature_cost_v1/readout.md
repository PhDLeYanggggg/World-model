# Frozen Interaction Features For Intervention Costs

## Material Passport

Fresh_run36real Torch cost-head fits and complete held SOURCE inference,
evaluation and replay;cached_verified frozen forecasters and h64cache.
This report reconstructs public scalar evidence,not new inference.

## Registered Primary

Interaction64minuszero64direct paired lower: **-0.0213 [-0.6148, +0.6482] pp**.
Known-label effect: +0.4800 [-0.1007, +1.1025] pp.
Upper effect: +0.9814 [+0.2950, +1.7400] pp.
Primary supported under the registered SOURCE bootstrap: False.
Original full-risk pass: False.

The two new508-input heads have identical33349parameter capacity,initial
weights,fit-only normalizer,full query draws,objective and2%policy rule.
Zero64removes the final standardized feature block exactly;the other
uses ego h64from a frozen producer-excluded EqMotion interaction layer.
The old444-input direct control has29253parameters and is secondary.

h64uses the original forecaster neighbor packet;the444descriptors use
the later partial-neighbor packet. This contrast tests the full added
feature bundle,not learned encoding isolated from raw neighbor coverage.
A positive result still needs a raw-original-context control before
that stronger attribution. This caveat was fixed before head training.

|Policy|Known gain % [95% CI]|Individual completion lower % [95% CI]|Supported/444|Worst signed easy net %|
|---|---:|---:|---:|---:|
|floor|+0.0000 [+0.0000, +0.0000]|+0.0000 [+0.0000, +0.0000]|0/444|0.0000|
|constant_velocity_causal_fd|+2.7513 [-4.9127, +11.1942]|-14.2417 [-25.5038, -3.6692]|0/444|20.8576|
|full_neural|+15.2748 [+11.1771, +19.5995]|+2.8440 [-12.3376, +13.2601]|0/444|235.8240|
|direct_bound_bounded|+3.4304 [+2.3220, +4.6454]|+2.7669 [+1.6691, +3.9250]|108/444|43.6929|
|zero64_bounded|+2.8701 [+1.9605, +3.9027]|+2.3436 [+1.4614, +3.3223]|93/444|39.4047|
|interaction64_bounded|+3.3501 [+2.0623, +4.8286]|+2.6165 [+1.3359, +4.0591]|110/444|39.3712|
|zero64_binary|+1.6270 [+0.9264, +2.4695]|+1.3737 [+0.7163, +2.1531]|29/444|33.9240|
|interaction64_binary|+2.1615 [+1.0070, +3.5191]|+1.8251 [+0.7784, +3.1037]|23/444|40.3990|

## Accuracy And Tail Diagnostics

|Policy|Known ADE pixel [95% CI]|Available endpoint FDE diagnostic pixel [95% CI]|Known gain over CV % [95% CI]|Mean role-specific p95 ADE pixel [95% CI]|
|---|---:|---:|---:|---:|
|floor|23.7721 [19.3426, 28.6125]|not estimable|-5.7369 [-16.8497, +3.9809]|86.1313 [67.7174, 106.8654]|
|constant_velocity_causal_fd|23.0342 [18.3989, 28.3641]|not estimable|+0.0000 [+0.0000, +0.0000]|81.0136 [61.8437, 105.3806]|
|full_neural|20.2433 [16.1795, 24.4996]|not estimable|+10.8141 [+0.2802, +18.4881]|74.1556 [59.0241, 91.6253]|
|direct_bound_bounded|22.9044 [18.6658, 27.4190]|not estimable|-2.0920 [-13.1790, +7.5340]|83.6979 [66.3535, 102.9301]|
|zero64_bounded|23.0552 [18.7470, 27.6203]|not estimable|-2.7623 [-14.2017, +7.0635]|84.4304 [66.5415, 104.3466]|
|interaction64_bounded|22.8655 [18.7057, 27.2741]|not estimable|-2.1850 [-13.3947, +7.4885]|83.2781 [66.3386, 102.0130]|
|zero64_binary|23.3598 [19.0130, 28.0180]|not estimable|-4.0875 [-15.5827, +5.9705]|85.7194 [67.6257, 105.9858]|
|interaction64_binary|23.1600 [18.9168, 27.6689]|not estimable|-3.5109 [-15.0125, +6.4527]|84.7141 [67.3113, 103.8138]|

The tail column is an equal-locality average of role-specific95th percentiles,
not a pooled-window percentile. Endpoint FDE uses only released valid
endpoints,not full-roster endpoint accuracy. The floor is producer-fit-
selected;CVremains a separate fixed causal comparator,not held-selected.

CV-relative percentages are averaged within localities before the final
mean;they are not the ratio of the two macro pixel-ADEaverages.

A paired lower comparison is not the subtraction of two individual lower
bounds. An empty selected reference is not estimable;the unchanged floor
is not thereby operationally unsafe or a learned-intervention success.

## Full Risk And Subsets

|Policy|Known all/easy violations|Completion all/easy violations|Undefined easy|Active %|Hard gain % [95% CI]|Easy gain % [95% CI]|
|---|---:|---:|---:|---:|---:|---:|
|direct_bound_bounded|143/235|174/332|4|57.9310|+2.4785 [+1.3477, +3.7382]|+4.2604 [+1.0898, +7.4824]|
|zero64_bounded|113/263|167/346|5|56.2847|+2.0506 [+1.2247, +2.9764]|+4.5495 [+1.5638, +7.7414]|
|interaction64_bounded|109/229|151/330|4|55.0101|+2.5600 [+1.3149, +4.0437]|+4.2613 [+1.2937, +7.3955]|

## Secondary Contrasts

interaction_minus_direct: known -0.0803 [-0.5621, +0.4403] pp;direct paired lower -0.6425 [-1.1074, -0.0951] pp.
zero_minus_direct: known -0.5604 [-0.9685, -0.2004] pp;direct paired lower -0.9658 [-1.3640, -0.6004] pp.

Secondary results cannot replace a failed registered primary.

## Fitting And Locality Evidence

|Feature arm|Mean full-bound fitting objective|Mean known-only fitting objective|
|---|---:|---:|
|zero64|1.498058|1.008933|
|interaction64|1.326343|0.904304|

These common decoded fitting objectives use the actual cyclic query visit
weights;they are not held evaluation or raw-FDE validation losses.

|Locality|Primary known pp|Primary paired lower pp|
|---|---:|---:|
|eu-locality-007|+0.6758|+0.0512|
|eu-locality-008|-0.4081|-0.7144|
|eu-locality-020|+0.3836|+0.0448|
|eu-locality-048|-0.1161|-0.3455|
|eu-locality-067|+0.1766|-0.0647|
|eu-locality-074|-0.9326|-1.1292|
|eu-locality-082|+0.9679|-1.2442|
|eu-locality-110|+1.8699|+1.8199|
|eu-locality-112|+0.0830|-0.4612|
|eu-locality-119|+3.0040|+2.5143|
|eu-locality-124|+1.0125|+0.6840|
|eu-locality-126|-0.9560|-1.4111|

## Endpoint And Cohort Support

Posthoc descriptive label-support counts,not a new primary cohort.
All six producer/seed views must agree on the same target-row support.

|Locality|Unique windows|Wholly unknown future|Complete12|Valid final endpoint|
|---|---:|---:|---:|---:|
|eu-locality-007|1089|45|275|515|
|eu-locality-008|81958|1860|49396|61617|
|eu-locality-020|415|12|143|228|
|eu-locality-048|1470|50|861|983|
|eu-locality-067|3631|75|1946|2525|
|eu-locality-074|55773|1001|36122|43522|
|eu-locality-082|78|16|0|0|
|eu-locality-110|1060|11|690|878|
|eu-locality-112|506|20|170|316|
|eu-locality-119|1658|30|934|1193|
|eu-locality-124|281|12|128|148|
|eu-locality-126|1591|31|944|1240|

082has zero released final endpoints in the unchanged whitelist;the
all-required-locality endpoint interval is therefore not estimable.
Do not fill endpoints,drop082or call available-labelADEa full12score.
008and074contain most windows,but the registered bootstrap gives each
locality equal weight. Window count alone is not independent support.

## Reproduction And Scope

Training PID52378:454.824s phase time including preparation/scoring,peakRSS12,552,601,600bytes.
Primary updates:116160;first full refits:4000;pilot:1200.
Complete replay PID55947:138.873s,including83.618s held prediction replay.
All36cost/amplitude pairs and72groups reproduce exactly;the old direct
policy matches every original metric view. Atomic checkpoint/resume,
arm64CPU4/inter-op1/workers0. See protocol and frozen registrations.

149510overlapping SOURCE windows,74recordings and12exposed localities.
444recording/pipeline views repeat those recordings,not444independent
scenes. Three actual seeds17/29/43;3000paired locality bootstraps after
role/seed averaging. Scoring sites are excluded from both producer and
critic fits,but SOURCE has been used for development,not final confirmation.

The source-cache v1pilot failed from neighbor-packet mismatch;v2restores
the original packet and separately replays all coordinates and h64rows.
The initial head pilot also failed before any optimizer update:the
wrapper requested Hrather than G,and the site assertion stopped it.
Temporary Hlabel preprocessing had been computed and discarded;
it was never used to train a head or select a policy. The explicit
registration_v2correction forwards the role and checks it earlier.
No private calibration/confirmation/DroneCrowd use for this repair.
Future targets and masks are loss/evaluation only. Available-label ADE
is not full12ADE;unknown-row bounds do not complete partial futures.
Pixels/rawstride12/silver,not metric,seconds,human gold or physical safety.
No new trajectory/JEPA/Transformer training,no deployment,Stage5C orSMC.
Licensed private data and checkpoints are needed for full raw reruns;
the public scalar reconstruction is not a standalone cold-start package.
