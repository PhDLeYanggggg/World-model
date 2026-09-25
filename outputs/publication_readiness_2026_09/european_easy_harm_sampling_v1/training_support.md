# Training-Source Easy-Harm Exposure

Result source: fresh_run B-only target and fitted-prediction diagnosis;
cached_verified preceding mean heads. No source-C outcomes are used here.
The parent closeout's 60 artifacts and 83 bindings were checked before this
audit. The detailed 36 group receipts are in training_support.json.

| Pair | Positive easy-harm examples per uniform batch of 256 | Median top 1% share of easy-harm mass | Median fitted / actual population easy-harm mass |
|---|---:|---:|---:|
| Full | 4.7463-8.5127 | 89.6497% | 0.7784 |
| Motion-only | 0.6501-2.1107 | 100% | 1.0278 |

The diagnostics reuse three B rosters across producer assignments and seeds.
They are dependent views, not 36 independent populations. Population agreement
does not imply agreement on the policy-selected subset or unseen localities.

Easy-harm mass is highly concentrated and batch exposure is low. This motivates
one controlled sampling test, not a claim that sampling is the sole cause of
underprediction. Ordinary oversampling and occurrence/severity decompositions
already have negative results elsewhere in this repository; they are not novel
or untested ideas. This study keeps the expected objective fixed, unlike a
naive change of class weights.

The protocol samples half the unchanged equal-locality distribution and half
its train-label easy-harm-weighted counterpart, then multiplies per-row loss by
the exact p/q ratio. A locality with no easy harm retains its original sampling.
Unknown future labels are not sampled. These labels are training supervision
only, never inference features. Existing B preprocessing and loss scales stay
fixed. No risk threshold, architecture or source role changes.

This is 8 observed / 12 predicted annotation steps at raw stride 12, image
pixels and detector-derived labels. It is not metric, seconds-level, human
gold, physical safety, true 3D, foundation-model or submission-ready evidence.
The six opened selection localities are not evaluated, and all 12 reserved
calibration plus six confirmation localities remain closed. Stage5C and SMC
remain off.
