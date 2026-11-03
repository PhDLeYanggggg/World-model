# Observation-Support Cost Learning On SOURCE

## Material Passport

Fresh_run36full real Torch cost-head fits, actual recovery/full refits,
held cost/support/amplitude inference, policy scoring and separate replay.
Cached_verified EqMotion forecasts and frozen direct-bound comparator.
This report uses public scalar receipts,not new training or inference.

## Registered Comparisons

Primary identified-minus-direct paired completion lower: **-3.1333 [-4.4103, -1.9871] pp**.
Paired known-label effect: -2.5648 [-3.7409, -1.4800] pp.
Within-new-family identification lower: -0.6745 [-1.0361, -0.3488] pp.
Marginal-family minus direct lower: -2.7861 [-3.8799, -1.7891] pp.

The primary versus the old direct head is an architecture/objective family
comparison,not isolated loss/capacity matching. Only the two new arms
match architecture, initialization, normalization and complete query draws.
A secondary comparison cannot replace the registered primary. A negative
completion-lower effect is not automatically observed-ADE regression.
Here the separate known-label interval is also below zero: the observed
utility regression is supported under this exposed-SOURCE bootstrap.

|Policy|Known gain % [95% CI]|Individual completion-lower % [95% CI]|Supported/444|Worst signed easy-net %|
|---|---:|---:|---:|---:|
|floor|+0.0000 [+0.0000, +0.0000]|+0.0000 [+0.0000, +0.0000]|0/444|0.0000|
|full_neural|+15.2748 [+11.1771, +19.5995]|+2.8440 [-12.3376, +13.2601]|0/444|235.8240|
|direct_bound_bounded|+3.4304 [+2.3220, +4.6454]|+2.7669 [+1.6691, +3.9250]|108/444|43.6929|
|marginal_support_bounded|+1.2279 [+0.9415, +1.5375]|+0.9568 [+0.6920, +1.2748]|109/444|34.2933|
|identified_support_bounded|+0.8657 [+0.6003, +1.1592]|+0.6051 [+0.2070, +0.9698]|160/444|8.3196|
|marginal_support_binary|+0.3173 [+0.1664, +0.5012]|+0.2601 [+0.1192, +0.4400]|29/444|35.3011|
|identified_support_binary|+0.2176 [+0.0984, +0.3730]|+0.1337 [-0.0841, +0.3307]|58/444|6.7019|

Individual lower bounds must not be subtracted to replace a direct
paired contrast. The zero-action floor adds no prediction error but has
empty selected reference support;0/444does not make it operationally unsafe.

## Complete Risk And Subsets

|Bounded policy|Known all/easy violations|Completion all/easy violations|Undefined easy|Active %|Hard gain % [95% CI]|Easy gain % [95% CI]|
|---|---:|---:|---:|---:|---:|---:|
|direct_bound_bounded|143/235|174/332|4|57.9310|+2.4785 [+1.3477, +3.7382]|+4.2604 [+1.0898, +7.4824]|
|marginal_support_bounded|6/211|30/331|4|52.3437|+0.7259 [+0.4978, +0.9895]|+3.4390 [+1.6886, +5.5224]|
|identified_support_bounded|14/199|35/280|4|52.8473|+0.4963 [+0.2906, +0.7071]|+2.8039 [+1.6113, +4.0353]|

All original2%all/easypositive-harm,signed easy-net and finite-support
requirements remain. No changed threshold, epsilon intervention, selected
seed or favorable locality rescues a failure. Full-policy pass is
**false**; registered primary support is
**false**.

Equal-locality mean amplitudes: marginal_support_bounded=0.122524, identified_support_bounded=0.091445, direct_bound_bounded=0.239083.
Active counts alone therefore do not describe the intervention change.
Composition and identification attenuate many surviving interventions.

## Fitting And Availability Diagnostics

|New arm|Full-bound fitting objective|Known-only fitting objective|Held unknown-support AUROC [95% CI]|AUPRC [95% CI]|Brier [95% CI]|ECE [95% CI]|
|---|---:|---:|---:|---:|---:|---:|
|marginal_support|1.497919|1.033432|+0.5702 [+0.5219, +0.6252]|+0.1202 [+0.0606, +0.2198]|+0.0471 [+0.0288, +0.0764]|+0.0488 [+0.0296, +0.0786]|
|identified_support|1.488498|1.022249|+0.6368 [+0.6041, +0.6701]|+0.1339 [+0.0756, +0.2221]|+0.0436 [+0.0248, +0.0715]|+0.0410 [+0.0216, +0.0701]|

Fitting objectives are complete common decoded scores with actual query-
visit weights,averaged across18jobs per arm. They are not the unlike
multitask training totals. Availability diagnostics predict wholly missing
released-label support,not baseline failure,physical visibility or complete
twelve-step labels. BCE is unweighted by class; its better empirical score
would still not establish calibrated selected-record risk. No comparison
to the older forest AUROC is treated as a matched feature effect.

Posthoc paired availability-AUROC diagnostic,identified-minus-marginal: +0.0666 [+0.0366, +0.0944].
This descriptive secondary target is not a trajectory/risk success and
does not replace the registered primary. It predicts released labels only.

## Locality Effects

|Locality|Primary known pp|Primary paired lower pp|
|---|---:|---:|
|eu-locality-007|-5.2932|-6.5224|
|eu-locality-008|-3.5217|-4.0302|
|eu-locality-020|-3.2812|-3.5233|
|eu-locality-048|-1.4468|-1.6804|
|eu-locality-067|-1.1770|-1.5415|
|eu-locality-074|-0.4169|-0.6134|
|eu-locality-082|-0.7193|-1.1258|
|eu-locality-110|-5.5744|-5.8015|
|eu-locality-112|-0.4518|-1.3987|
|eu-locality-119|-5.1886|-6.0593|
|eu-locality-124|-3.5198|-4.5132|
|eu-locality-126|-0.1868|-0.7897|

## Runtime And Roles

Primary updates:116160; full-refit:4000; pilot:1200.
Fitting PID17236:404.408s,peak9,229,516,800bytes.
Separate verification PID18551:131.106s,including77.438s complete cost/support/amplitude inference.
All36prediction triplets and72metric groups replay exactly. Nativearm64
Torch CPU4/inter-op1/workers0,atomic checkpoints,heartbeat and resume.
The old direct policy reproduces exactly on every required view.

149510SOURCEwhitelist windows,74recordings,twelve exposed localities.
Each evaluation locality is excluded from both forecaster and critic fits.
Three actual seeds17/29/43;444recording views are repeated,not444independent
scenes. CIs use3000locality resamples after excluded-view/seed averaging.
These are development findings,not independent final confirmation.

No private calibration,confirmation or DroneCrowd arrays. Targets/masks
are supervision/evaluation only; no future endpoint,central velocity,test
endpoint goals,eventual track length or future masks in inference.
Available-label ADE is not full12ADE; bounds do not fill partial futures.
Pixels/rawstride12/released silver are not metric,seconds,human gold,
physical safety,true3D or foundation. New models are cost heads,not new
trajectory/JEPA/Transformer dynamics. No deployment,Stage5C orSMC.
