# Causal Future-Label-Support Probe

## Material Passport

Fresh_run18 full ExtraTrees fits, a complete first-model refit, real
recovery pilot and full prediction/metric replay. Cached_verified SOURCE
arrays/geometry/detector-history/role definitions. This report itself reads
public scalar evidence, not raw data or fresh model inference.

## Complete Held-SOURCE Results

|Arm|AUROC [95% CI]|AUPRC [95% CI]|Unknown prevalence|AUPRC minus prevalence [95% CI]|Brier|ECE|
|---|---:|---:|---:|---:|---:|---:|
|geometry|0.6023 [0.5395, 0.6565]|0.0798 [0.0556, 0.1085]|0.0417|0.0380 [0.0236, 0.0534]|0.0427|0.0300|
|geometry_detector|0.7062 [0.6481, 0.7508]|0.1440 [0.0948, 0.2177]|0.0417|0.1023 [0.0695, 0.1468]|0.0398|0.0294|

Registered detector-minus-geometry AUROC effect: 0.1039 [0.0600, 0.1622].
The primary is retained even if geometry alone has a better result.
The unknown-prevalence value is an evaluation reference, not a held-label
probability fed into inference. No probability calibration or threshold
selection is fitted on held outcomes. All point estimates are equal-locality
means, including prevalence; they are not pooled row-level event rates.
Scores need not be well calibrated.

## What This Tests

The positive event is that none of twelve requested future labels exists
in the release, not physical occlusion. Past geometry476 and detector64
are causal cached-verified fields. The future mask supplies supervision
and evaluation labels only; no mask, endpoint, eventual track length,
central velocity or test statistic enters the feature matrix.
Each forest fits its original four SOURCE localities and predicts the
excluded eight. There are149,510windows,74recordings,twelve localities.
The nine seed/fold views per arm repeat data, not independent people.
Intervals use3,000locality resamples after averaging two excluded folds
and actual forest seeds17/29/43. SOURCE has informed research choices;
this is crossfit development evidence, not independent confirmation.

## Fitting And Domain Breakdown

|Arm|Fitting AUROC min/max across all nine models|Held-SOURCE AUROC min/max across localities|
|---|---:|---:|
|geometry|0.7255/0.9221|0.3479/0.7244|
|geometry_detector|0.7873/0.9564|0.4491/0.8126|

Fitting scores are resubstitution on other sites, not paired held validation;
their apparent gap does not isolate one cause of domain shift.

|SOURCE locality|Geometry AUROC|Geometry+detector AUROC|Geometry+detector AUPRC|
|---|---:|---:|---:|
|eu-locality-007|0.5376|0.6649|0.1018|
|eu-locality-008|0.6619|0.7567|0.0758|
|eu-locality-020|0.4494|0.4491|0.0797|
|eu-locality-048|0.7078|0.7876|0.2087|
|eu-locality-067|0.6642|0.7748|0.1314|
|eu-locality-074|0.6581|0.7408|0.1054|
|eu-locality-082|0.3479|0.7324|0.5062|
|eu-locality-110|0.6201|0.6722|0.0573|
|eu-locality-112|0.6832|0.6813|0.1141|
|eu-locality-119|0.7244|0.8126|0.0993|
|eu-locality-124|0.5765|0.6753|0.1498|
|eu-locality-126|0.5970|0.7271|0.0988|

## Recording51 At Locality082

|Arm|Trial|Unknown rows / total|Unknown score mean|Known score mean|AUROC|AUPRC|
|---|---|---:|---:|---:|---:|---:|
|geometry|single1_seed17|16/78|0.0201|0.0290|0.2177|0.1702|
|geometry_detector|single1_seed17|16/78|0.0312|0.0288|0.6784|0.3267|
|geometry|single1_seed29|16/78|0.0201|0.0269|0.2545|0.1699|
|geometry_detector|single1_seed29|16/78|0.0307|0.0276|0.6925|0.3337|
|geometry|single1_seed43|16/78|0.0185|0.0244|0.2581|0.1783|
|geometry_detector|single1_seed43|16/78|0.0290|0.0252|0.7278|0.3372|
|geometry|single2_seed17|16/78|0.0629|0.0671|0.4607|0.2159|
|geometry_detector|single2_seed17|16/78|0.1583|0.0880|0.7601|0.6462|
|geometry|single2_seed29|16/78|0.0605|0.0698|0.4410|0.2290|
|geometry_detector|single2_seed29|16/78|0.1378|0.0880|0.7530|0.6262|
|geometry|single2_seed43|16/78|0.0716|0.0781|0.4551|0.2414|
|geometry_detector|single2_seed43|16/78|0.1417|0.0714|0.7823|0.7670|

No scene/recording is dropped, and one-class recording groups remain
not estimable rather than favorable perfect scores. A useful ranking here
would still not prove safe intervention or eliminate unknown-row bounds.

## Compute And Reproduction

Primary forest trees:2304;first full refit:128;pilot:128.
Training PID6433:267.972s,peak4,513,054,720bytes.
Complete prediction replay PID7299:14.840s.
All18probability arrays and144model/locality metric groups reproduce exactly.
Forest training uses four native threads, deterministic single-thread
probability reduction,per32-tree checkpoints and unchanged resume.
These are actual fitted forests, not new Torch dynamics or NumPy fallback.

```sh
.venv-pytorch/bin/python scripts/run_m3w_source_label_support.py train --resume
.venv-pytorch/bin/python scripts/report_m3w_source_label_support.py
```

Resume applies to unfinished work; complete stage receipts are immutable.
The original completed replay and public-scalar reconstruction are distinct.

## Limits And Next Action

This probe does not train a selector, use its probabilities for admission,
or pass a new2%recording-risk contract. The primary feature-ablation effect
must be separated from absolute label-support predictability. It cannot
establish neural dynamics, scene/goal contribution or causal physical visibility.
If signal is useful, the next experiment must compare explicit unknown-row
disagreement-bound supervision against ordinary known-only cost learning,
with calibration and final confirmation still closed. If it fails, diagnose
the past-input/label-process gap rather than giving future availability as input.
Pixels/rawstride12/released silver only; not metric,seconds,human gold,
physical safety,true3D or foundation. No deployment upgrade,Stage5C or SMC.
