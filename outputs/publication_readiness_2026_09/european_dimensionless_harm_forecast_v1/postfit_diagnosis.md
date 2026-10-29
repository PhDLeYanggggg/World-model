# Fixed-Endpoint Failure Diagnosis

Post-fit analysis of verified source-development results; no new model selection or threshold search.
All nine endpoints, all three seeds, all twelve localities and all 163 recordings retained.

## Matched Predictive Effect

Equal-locality ADE change: -0.929724 pp versus the original model; conditional 95% locality-bootstrap CI [-1.6693101778828632, -0.16849728614315282].
Localities improved: 4; worsened: 8.
Seed-specific ADE changes (pp): {'17': -0.8982204190734783, '29': -0.6678456499128925, '43': -1.2231056424820406}.

## Known Recording-Level Risk

The 978 views repeat 163 recordings across two excluded producers and three seeds.
These are unrestricted forecast diagnostics, not the final intervention policy or safety certification.
Missing outcomes are not completed as zero. Known net benefit cannot cancel positive harm.

| Arm / subset / reference | Views | Known <=2% | Known >2% | Not estimable | Net degradation >2% | Views with missing labels |
|---|---:|---:|---:|---:|---:|---:|
| new/all/CV | 978 | 15 | 963 | 0 | 140 | 978 |
| new/all/selected | 978 | 567 | 411 | 0 | 17 | 978 |
| new/complete/CV | 978 | 31 | 941 | 6 | 119 | 0 |
| new/complete/selected | 978 | 595 | 377 | 6 | 25 | 0 |
| new/easy/CV | 978 | 56 | 922 | 0 | 536 | 0 |
| new/easy/selected | 978 | 622 | 356 | 0 | 74 | 0 |
| new/hard/CV | 978 | 37 | 932 | 9 | 20 | 0 |
| new/hard/selected | 978 | 516 | 453 | 9 | 28 | 0 |
| old/all/CV | 978 | 0 | 978 | 0 | 146 | 978 |
| old/all/selected | 978 | 13 | 965 | 0 | 36 | 978 |
| old/complete/CV | 978 | 16 | 956 | 6 | 117 | 0 |
| old/complete/selected | 978 | 27 | 945 | 6 | 51 | 0 |
| old/easy/CV | 978 | 7 | 971 | 0 | 590 | 0 |
| old/easy/selected | 978 | 17 | 961 | 0 | 405 | 0 |
| old/hard/CV | 978 | 13 | 956 | 9 | 14 | 0 |
| old/hard/selected | 978 | 60 | 909 | 9 | 18 | 0 |

## Training Diagnostics

Same scheduled minibatches; last ten logged batches only, not whole-TRAIN or validation loss.
Compare the native ADE term, not total objectives with different penalty definitions.

| Fit | Original native ADE | New native ADE | New selected harm | New CV harm |
|---|---:|---:|---:|---:|
| single0_seed17 | 0.876368 | 0.914298 | 0.032424 | 0.070357 |
| single0_seed29 | 0.885893 | 0.902995 | 0.027444 | 0.055308 |
| single0_seed43 | 0.983639 | 1.030892 | 0.036194 | 0.049639 |
| single1_seed17 | 0.924594 | 0.964174 | 0.017662 | 0.055292 |
| single1_seed29 | 0.993007 | 1.044917 | 0.029321 | 0.036688 |
| single1_seed43 | 0.884709 | 0.961087 | 0.023217 | 0.041743 |
| single2_seed17 | 0.743869 | 0.761708 | 0.018617 | 0.099031 |
| single2_seed29 | 0.840539 | 0.850517 | 0.018377 | 0.112105 |
| single2_seed43 | 0.880816 | 0.931537 | 0.020788 | 0.109234 |

## Fixed-Floor Support

Original 146/148; new 147/148; gained 1, lost 0.
The old oracle, frozen CV/damping.97 floor and original eligibility replay exactly.
The remaining unsupported view is retained. An oracle is not an inference policy.

The aggregate penalty reduces harm but sacrifices overall/hard prediction quality. 
It is not a recording-level relative-risk constraint. Its train-selected reference is also 
not the CV/damping.97 deployment floor. These facts motivate a training-only reference/constraint 
repair, but do not prove a unique cause or a general impossibility result.

Image-pixel/raw-frame detector-track evidence only. No independent confirmation, deployment, 
Stage5C, SMC, metric, seconds, human-gold, true3D or foundation claim.
