# Forecast Seed Disagreement As A Switch-Risk Covariate

## Material Passport

Mode:run and reproducibility validation. Scope:SOURCE-only model development,
then frozen-policy evaluation on the six already exposed model-selection sites.
Not independent risk calibration,confirmation,new dynamics or submission-ready evidence.

## Motivation And Fixed Hypothesis

The previous27-fit factorial loss experiment has a positive registered utility
contrast but worse risk:mean_absolute paired lower+.799527pp[.175107,1.333576],
worst known easy net8.495468%,support6/252repeated record views. The original
point-forecast critic sees one neural path but not variation among actual
training seeds. Test whether same-fold forecast disagreement adds useful harm
information without modifying forecasts,labels,split,risk budget or policy rule.

Deep ensembles are established,not a new M3W invention; see
[Lakshminarayanan et al.,2017](https://arxiv.org/abs/1612.01474) and
[Ovadia et al.,2019](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html).
Here the members are frozen point forecasters,not probabilistic likelihood heads.
Their spread is a covariate,not a predictive interval,posterior,calibration
guarantee or complete measure of epistemic/aleatoric uncertainty. Dataset shift
can defeat it. No deployment or novelty claim follows from implementing it.

## Feature Construction And Lineage

For each SOURCEfold,use actual seeds17/29/43 trained on the same four producer
sites. Their SOURCE cost rows are the identical eight excluded localities and
must have identical row IDs/order. All three producer ancestries exclude those
rows' localities. On selection,all members are frozen SOURCE-trained models.
Do not combine producers that trained on a cost row's scene.

Construct16fields from the three relative predicted paths and original positive
past-only scale:12stepwise radial RMS spreads,max pair path disagreement,
endpoint direction dispersion,mean step-speed spread,total-path-length spread.
Predicted future rollouts are allowed; observed future endpoints,labels,validity
masks and central velocity are not. Freeze three SOURCE and three selection
feature arrays before training. Parent checks may hash supervised artifacts,
but feature construction never decodes or uses their outcome values.

SOURCE-only normalization is equal-locality mean/second moment,clip8. Noise is
a deterministic row-ID-hashed draw from an8x512row fitting-SOURCE descriptor
catalogue,balanced across the eight sites. It is a frozen fitted noise prior,
not a current-query observation or a draw from selection statistics. The same
catalogue/normalization is used across arms;no label selects catalogue entries.

## Three Matched Arms

disabled:zero added input and original forward exactly.
real:actual current-query forecast spread.
noise:fitting-SOURCE-matched random descriptor prior,independent of row outcomes.

Use the fixed mean_absolute cost loss from the preceding experiment. The old
444field context critic receives one additional16->64biasless zero-initialized
projection. All arms have30,277parameters and identical original weights,RNG,
zero projection,query draws,budgets,known-only labels and original coherent
B/H/R/ER/EH output constraints. No raw image,scene ID,goal or new trajectory
model is introduced. Only the uncertainty-input arm differs.

Fit all27endpoints,nine per arm. Engineering pilot:50->100resumed vs100direct
for all three arms;disabled100step must reproduce the old mean_absolute pilot.
Formal run:each first-producer arm gets a separate complete exact refit;every
disabled full fit must reproduce its original mean_absolute weights,base
optimizer states,trace,normalization,sampler and RNG. No tolerance relaxation.
Each original query has at least four visits;do not sample only favorable rows.

## Decision And Evaluation

Freeze all27new pointwise action policies before scoring this comparison.
The original gain>0,all/easy predicted positive-harm2%guards remain. No
threshold search or new joint allocator. Scores use all38,102targets/28records/
six localities,including partial and wholly unknown outcomes.

Primary:real minus disabled paired completion-lower utility versus the same
SOURCE-selected floor. Negative control:real minus noise,same metric. Report
known secondary,all arms,per-locality/seed effects,switch rate,all/easy positive
harm and worst easy net. Bootstrap3000localities after averaging producerfolds
and seeds;overlapping windows and repeated views are not independent samples.
Advancement requires both paired-lower CI lower bounds>0 and original full-
record risk support,including unchanged2%all/easy harm and easy-net budgets.
An empty selected-easy reference remains not estimable,not a safety pass.
Do not replace the failed earlier matched-joint primary with this input test.

Selection is exposed development,not final test. No selection outcome fits
weights,normalization,noise priors or thresholds. Calibration/confirmation/
DroneCrowd remain closed. SOURCE fitting performance is resubstitution,not
validation. Keep all negative results and run separate inference/metric replay.

## Runtime And Claim Boundary

Local arm64Torch,CPU4/inter-op1/workers0;100step atomic checkpoint/heartbeat,
same-contract resume,12hper-process boundary,10GiBdisk reserve. Healthy local
training is allowed;no automatic quick downgrade. No CREATE submission unless
actual resource evidence justifies it. Private data/forecasts/features/weights
stay outside Git;only code/configs/scalar reports/derived figures are public.

obs8/pred12/rawstride12,local-image pixel,detector-silver,available-label ADE.
Not old rawt50,metric,seconds,human gold,true3D or foundation evidence.
Stage5C/SMC remain off. Current deployed model is unchanged at registration.
