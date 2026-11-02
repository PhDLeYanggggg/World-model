# SOURCE Cost-Fit Diagnosis: Complete

Fresh_run: frozen Torch inference on all18cost states and separate complete
replay. Cached_verified: weights, original SOURCE inputs and SOURCE-only
normalization. All144site groups reproduce exactly, including all72original
control groups. No training, policy or threshold change. These SOURCE rows
were used to fit the critics: this is resubstitution, not validation.

## Selected Record Medians

|Cohort/family|H predicted/true|EH predicted/true|R predicted/true|ER predicted/true|
|---|---:|---:|---:|---:|
|SOURCE / weighted_centered|1.25781|0.00257925|3.51803|0.114322|
|SOURCE / mean_absolute|0.365326|0.000503009|1.88242|0.0416106|
|Calibration / mean_absolute independent lambda1|0.201091|2.96084e-05|1.40482|0.0115603|

Medians condition on defined ratios; finite counts and unknown rows
remain in the JSON. SOURCE and calibration rosters differ; this is not a
paired transfer effect, a causal shift decomposition or a new operating-point
selection. The calibration row uses the same independent policy definition.

## SOURCE Pooled Cost Masses

|Family/pool|H predicted/true|EH predicted/true|Known all risk %|Known easy risk %|
|---|---:|---:|---:|---:|
|weighted_centered/all_legal|1.65597|0.3525|31.7793|38.7745|
|weighted_centered/selected|1.13763|0.0117858|3.04386|7.38791|
|mean_absolute/all_legal|0.819612|0.052092|31.7793|38.7745|
|mean_absolute/selected|0.281149|0.00364461|5.10903|11.4495|

These pool repeated action views, not independent people or recordings.
All-legal candidate costs are not a deployment policy. Known-label risk
does not replace original unknown-completion and finite-support checks.

## Conclusion And Controlled Next Step

The mean-cost repair has severe selected easy-harm error even on fitting
SOURCE. New-locality amplification is also present. A pure domain-shift
explanation is therefore insufficient. This does not prove that a particular
loss or activation is the sole cause: selection, sparse supervision and
finite representation/capacity remain plausible mechanisms.

The historical stable easy-deviance experiment already failed. Its code
uses stable logs; do not invent a clipping bug or call that method new.
A narrow next test adapts the existing stable cost divergence to the current
three-forecaster-seed mean_absolute framework, with easy-only and H+EH
controls. Same model, complete query coverage, draws, initialization,
SOURCE scope and 2% evaluation remain. Any subsequent selection-cohort
readout is exposed development, not independent confirmation.

## Runtime

|Phase|PID|Seconds|Peak RSS bytes|
|---|---:|---:|---:|
|pilot|28699|7.505672|3939713024|
|complete|28953|65.986890|8706785280|
|replay|29316|72.032816|8681799680|

The primary run reuses two hash-checked current-run pilot cases and
freshly computes sixteen; separate replay freshly computes all18. No large
array or model is written by this diagnosis. Local native arm64CPU4, workers0.
No CREATE job submitted. Calibration arrays, confirmation and DroneCrowd
remain unopened by this operator. No deployment promotion or risk certificate.
Obs8/pred12/rawstride12, local-image pixels, detector/tracker silver; available
label ADE is not full12 ADE. Not metric/seconds, human gold, true3D or foundation.
Stage5C and SMC remain off.
