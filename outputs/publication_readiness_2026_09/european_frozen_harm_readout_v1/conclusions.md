# Frozen Harm Readout: Completed, Not Promoted

## Material Passport

This is source-development evidence, not independent confirmation. Parent
forecasts, encoders, labels and original controls are cached_verified.
The membership decomposition, 288 readout fits, predictions and paired
statistics are fresh_run. New trajectory training, policy evaluation,
independent calibration and confirmation are not_run by design.

Registration `fa128423` preceded fitting. Prediction freeze `cf479a70`
preceded the outcome readout. All six source assignments, three seeds,
two forecast pairs and four held localities per assignment are retained.
Both arms received the same 2,000 updates and sampled rows. Their reference
cost predictions remain exactly equal to the original mean head.

## Main Answer

The frozen fractional representation does not pass the expected-harm
magnitude test. Compared with an equally trained frozen-mean readout, all
six full-pair conditional easy-harm MSE intervals overlap zero. Compared
with the original mean head, three intervals are negative and three overlap
zero. The registered advancement gate fails. Deployment is unchanged.

Positive numbers below mean lower conditional easy-harm MSE, not lower ADE
or FDE. Each interval uses 3,000 locality resamples after averaging three
seeds within locality. Only four localities enter each assignment; repeated
assignments/windows are not independent and intervals are exploratory.

| Source roles | Fractional vs matched gain % [95% CI] | Fractional vs original gain % [95% CI] |
|---|---:|---:|
| producer0 / controller1 | -0.148 [-1.823, 0.966] | -1.832 [-2.937, -0.214] |
| producer0 / controller2 | -0.140 [-5.766, 5.225] | -39.789 [-98.368, -0.499] |
| producer1 / controller0 | -0.043 [-1.207, 1.633] | -2.165 [-4.173, -0.157] |
| producer1 / controller2 | 0.525 [-0.571, 2.012] | -3.334 [-11.071, 0.860] |
| producer2 / controller0 | 2.373 [-0.207, 5.707] | 1.411 [-2.673, 5.495] |
| producer2 / controller1 | -9.357 [-24.613, 0.129] | -21.795 [-64.376, 0.519] |

The matched-mean refit is not an improvement either: its six role-level
point gains against the original head are all negative, from -44.421%
to -1.001%, with one negative interval. A weaker matched control must not
be used to present the fractional representation as a successful repair.

## Secondary Findings

The full fractional-feature readout retains some ranking information:
conditional AUROC median is 0.56228 versus 0.47697 for the matched control;
five paired intervals are positive. Conditional top-10% harm capture has
three positive paired intervals and three overlaps. Coverage-error reduction
has one positive interval. These do not override the primary failure.
The original fractional head's conditional AUROC was 0.61971: the cost-only
readout does not preserve all of its earlier ranking performance.

Full median predicted/actual easy-harm coverage moves from 0.66266 for
original_mean to 0.97504 for mean_features and 1.03000 for
fractional_features. Better average coverage is not better per-row cost
prediction. Motion-only is worse: fractional-feature MSE has two negative
intervals against its matched readout and four against original_mean.

## Mechanism and Next Step

Fractional features improve training easy-harm MSE in 62/72 dependent full
views relative to matched mean features, but improve held MSE in only 39/72.
Twenty-seven views improve only in fitting. Against original_mean, the
fractional readout worsens 42/72 full views; outside-easy rows dominate the
excess error in 36 of these 42. This repair has not resolved easy-membership
spillover or transport of harm magnitude.

Next test the learnability and locality transport of the easy-membership
label itself, separately from harm within easy cases. Any follow-on fit
must distinguish that label from the previously tested positive-harm hurdle,
keep future error strictly in supervision, and retain matched controls.
Do not retune intervention thresholds on these outcomes or open a C policy
readout with a failed risk model.

## Verification and Scope

All 576,000 real Torch updates completed, with zero unknown-label draws and
184.69 summed fitting seconds. The main training process's first fit to last
head-freeze timestamps span 448 seconds; this excludes initial startup,
pilot and subsequent evaluation/verification. All 288 checkpoint prefix
replays and 5,184 independent rank/tail metric checks pass. Each replay
compares up to 4,096 held predictions; it is not an assertion of complete
raw-video reprocessing. Reference moments are compared over all held rows.
Final artifact hashes, test count and scope are recorded in `verification.json`.
Loss and CI figure previews were visually inspected.

Eight observed / twelve predicted annotation steps, raw stride 12,
detector-derived image pixels. No metric/seconds, human-gold, physical
safety, true 3D, foundation or submission-ready claim. Reserved calibration
and confirmation remain closed. Stage5C and SMC remain off.
