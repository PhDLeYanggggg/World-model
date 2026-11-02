# Cardinal Cost-Loss Repair: Complete Readout

## Scope And Result Sources

Fresh_run:27complete Torch SOURCE cost fits,three separate complete first-producer
refits,new causal cost inference,27pointwise policy freezes,and54selection metric
groups. Cached_verified:the nine original context critics,the original fixed
forecast bank, SOURCE rows and exposed selection labels. No new forecasting
model,data split,normalization,feature,architecture or risk threshold is introduced.
The six selection localities are already exposed development data,not final test.
Calibration/confirmation/DroneCrowd remain closed. SOURCE resubstitution is not validation.

All arms share29,253parameters,initial weights,complete query draws and original
producer-excluded eight SOURCE sites. The forecaster fits the other four sites.
Each source query has at least four visits. Unknown outcomes are not supervised
zeros. Inference accepts only the original444causal fields and causal rollouts.

## Controlled Factors

Original weighted_centered has H/EH underprediction multiplier4 and within-row
centered net-gain-error loss. mean_centered changes only multiplier4to1;
weighted_absolute changes only centered to uncentered net-gain error;
mean_absolute combines both. The latter is the registered primary,not selected
after looking at outcomes. Same decoder shares R/ER and keeps B<=R,ER<=R,EH<=H
and B+H<=causal disagreement. Removing asymmetry does not guarantee correct
conditional means in this clipped,finite-capacity,multi-objective model.
Absolute gain anchoring is not a new calibration guarantee or a new dynamics model.
The historical144-fit easy-deviance failure remains a distinct negative result.

## Registered Primary And Factor Controls

All contrasts versus the frozen weighted_centered context independent policy;
units are percentage points relative to each producer's SOURCE-selected floor.

|New loss arm minus original|Known-label secondary [95% CI]|Paired completion-lower [95% CI]|
|---|---:|---:|
|mean_absolute|+1.3443 [+0.7178, +1.8950]|+0.7995 [+0.1751, +1.3336]|
|mean_centered|+0.5246 [+0.2312, +0.8863]|+0.0455 [-0.1882, +0.2704]|
|weighted_absolute|+0.4439 [+0.1751, +0.7531]|+0.2294 [-0.0721, +0.5533]|

Primary pass:**True**. Full-policy original record-risk pass:
**False**. Candidate:**False**.
Deployment changed:**false**. Positive secondary cannot replace the primary or
the unchanged all/easy positive-harm requirements. No joint solver is retrained
or reoptimized; the earlier failed matched-joint scientific result remains.

## Accuracy, Intervention And Safety

|Policy|Known ADE gain vs floor,% [CI]|Mean switch rate|Worst known easy net degradation|Supported repeated record views|Known all/easy risk violations|Undefined easy-risk views|
|---|---:|---:|---:|---:|---:|---:|
|mean_absolute|+1.5037 [+0.8170, +2.1307]|22.496%|8.495%|6/252|199/211|14|
|mean_centered|+0.6841 [+0.3057, +1.1437]|12.853%|9.232%|21/252|184/188|25|
|weighted_absolute|+0.6033 [+0.2730, +0.9927]|11.091%|3.417%|19/252|154/179|28|
|weighted_centered|+0.1595 [+0.0580, +0.2726]|5.521%|3.772%|23/252|147/154|52|

All38,102targets,28recordings,six localities remain,including futureless and
partial outcomes. There are252repeated recording-risk views,not252independent
recordings. The2%budget applies to every required record's positive all/easy
harm and easy-net degradation. Benefits cannot cancel positive-harm numerators.
An empty/zero-reference easy-risk view is not estimable,not a safety pass.
No favorable subset or seed is removed. These are available-label ADE scores;
full12step FDE is not available on the entire roster. Endpoint scores are diagnostic.

## Primary By Locality

|Selection locality|Known component,pp|Paired completion lower,pp|
|---|---:|---:|
|eu-locality-087|+1.9690|+1.4960|
|eu-locality-092|+0.4009|+0.0855|
|eu-locality-093|+1.5148|+0.8562|
|eu-locality-103|+1.6526|+1.2273|
|eu-locality-104|+0.2378|-0.4624|
|eu-locality-125|+2.2905|+1.5945|

Intervals:3000locality bootstrap draws,after averaging three SOURCE folds and
actual seeds17/29/43 inside each locality. Overlapping windows and repeated
producers are not independent samples. This is conditional development evidence,
not calibrated population safety or untouched final confirmation.

## Actual Training Loss And Runtime

|Loss arm|Complete new fits|Primary updates|Mean first minibatch total|Mean final minibatch total|
|---|---:|---:|---:|---:|
|mean_centered|9|70656|2.74357|0.22627|
|weighted_absolute|9|70656|4.02793|0.30802|
|mean_absolute|9|70656|3.36773|0.27711|

Loss definitions differ by design; cross-arm raw training totals are not a
common test metric. First/final minibatches differ and are not validation loss.
Scalar per-update traces are exported separately from hash-verified checkpoints.
[All real fitting traces](figures/training_traces.svg) and
[paired utility and easy-net risk](figures/utility_and_easy_risk.svg) are derived
only from these public scalar artifacts,not new experiments.
No weights or labels are published. The original control100-step weights,
optimizer,sampler,trace and RNG match the previous real pilot exactly. Every
new arm passes50->100resume vs100direct; three complete first-producer refits
also match exactly. These are engineering reproducibility checks,not independent
scientific confirmation.

Formal train PID93797:622.392853s,peak14072610816B;
primary updates211968,complete-refit updates21504.
Evaluation PID94908:16.691139s. Separate replay PID95068:
29.080005s. All27new cost inferences,choices and54scalar metric groups
reproduce exactly; cached original control identity remains exact. Local arm64
CPU4/inter-op1/workers0. No CREATE job or remote scheduler claim.

## Interpretation Boundary

The result tests a specific SOURCE objective repair,not new JEPA/Transformer
representation or trajectory learning. Utility,positive harm,unknown outcomes
and support must be read together. Risk calibration and a useful safe joint
policy remain separate requirements before final confirmation. Do not deploy a
better mean with unsafe recordings or call a zero-switch fallback learned success.

obs8/pred12/rawstride12,local-image pixels,detector-silver,2.5D/pseudo-3D agent
trajectory state modeling. Not metric,seconds,human gold,true3D or foundation
evidence. No Stage5C/SMC. Submission readiness remains false.
