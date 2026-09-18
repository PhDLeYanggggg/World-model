# Removing Static-Target Gradients Does Not Repair Candidate Transfer

## Result and Decision

**Do not deploy this candidate and do not treat its oracle increase as learned
safe routing.** Twelve fresh cold-start neural models completed 120,000 updates.
Every training-complement and inner-held full-cohort gain is negative. Removing
zero-target ADE gradients gives larger candidate/CV oracle headroom but severely
increases actual error, including on nonzero-target rows.

| Quantity | Unconditional control | Zero-target-gradient intervention |
| --- | ---: | ---: |
| Equal-site ADE gain over stationary CV | -5.01598% | -98.71920% |
| Conditional four-site 95% interval | [-8.39655%, -2.48773%] | [-128.49923%, -66.02162%] |
| Future binary-oracle gain, diagnostic only | +0.46765% | +3.75968% |
| Nonzero-target window-weighted gain | -0.93253% | -32.38720% |
| Zero-target absolute ADE harm, annotation pixels | 0.09190 | 1.42880 |
| Full-cohort mean ADE, annotation pixels | 1.18919 | 2.28607 |

The unchanged stationary-CV mean ADE is 1.12765 annotation pixels. Primary
normalization and equal-site weighting differ from native-pixel pooling; these
columns must not be substituted for one another. A -98.71920% gain means that
the primary error is approximately 1.987 times the baseline, not a negative ADE.

Matched actual gain deteriorates by 93.70322 percentage points, conditional
interval [-120.10267, -62.45914]. Oracle gain increases by 3.29204 points,
interval [3.04645, 3.64157]. The oracle sees future labels and cannot be deployed.
Three-seed averaging is an average of errors, not a prediction ensemble.

## What Was Changed

The pre-fit registration, pushed as `ee3e8667`, retains the model, initial seeds,
all training IDs, uniform sampler, loss scale, fitted normalization, optimizer,
learning-rate schedule and terminal 10,000-update budget per model. The only
loss intervention is multiplication of entirely zero-target rows' ADE by zero.
The denominator remains the full batch size; nonzero rows are not oversampled.
The twelve final sampler states and per-row draw counts exactly match controls.

Future labels affect supervised training loss and evaluation only. They do not
select inference rows, inputs, checkpoints or a deployment policy. Every held
row is retained. AdamW state and weight decay still operate; removing one class
of gradient is not a pure experiment on a conditional-median theorem.

## Per-Site Outcomes

| Inner held site | Queries | Actual gain | Future-oracle gain | Hard gain | Zero-target pixel harm |
| --- | ---: | ---: | ---: | ---: | ---: |
| coupa | 4,250 | -101.20999% | +4.24115% | -4.13386% | 1.77329 |
| deathCircle | 3,054 | -53.19262% | +3.44298% | -1.36105% | 0.84821 |
| gates | 1,662 | -133.47483% | +3.43294% | -1.69448% | 1.16790 |
| hyang | 6,464 | -125.27185% | +3.81764% | -5.49321% | 1.69255 |

Hard cutoffs are fitted on the corresponding producer's training complement.
No site or seed is discarded. Of the 98.71920 points of primary excess error,
71.63337 arise on zero-target rows and 27.08583 on nonzero-target rows. Perfect
rejection of all static futures would therefore still not repair the average
remaining candidate. The fraction of nonzero-target rows benefiting from the
candidate falls from about 41.1% to 25.4%-26.9% across seeds.

## Direction Negative Control

The direction diagnostic was registered during training, after early raw
negative scores but before computing its results (`4e21587d`). It is explicitly
post-hoc, not part of the pre-fit comparison. Rotate each fixed relative path
by +90, -90 or 180 degrees, preserving its length, shape and bounds. No training,
target-dependent rotation choice or new policy is involved.

| Fixed direction, motion-loss candidate | Oracle gain | Original-minus-null, points | Conditional 95% interval |
| --- | ---: | ---: | --- |
| Original | +3.75968% | Reference | Not applicable |
| +90 degrees | +3.49561% | +0.26407 | [+0.00254, +0.45966] |
| -90 degrees | +3.58218% | +0.17750 | [-0.22349, +0.54639] |
| 180 degrees | +3.31027% | +0.44941 | [+0.27857, +0.58881] |

The original direction has a positive point contrast against all three nulls,
but one interval crosses zero. These are unadjusted descriptive contrasts on
four explored sites, not independent evidence of a stable directional method.
The substantial oracle gains of all rotated paths caution against interpreting
the original 3.76% as directional skill. They do not prove that direction is
irrelevant or causally quantify how much benefit comes from magnitude.
Zero-target harm is identical under rotations. The cached unconditional
candidate's original oracle is lower than all three rotated controls.

## Failure Taxonomy

1. **Objective/population mismatch, demonstrated:** removing static-target loss
   encourages candidates that are unsafe for the full inference population.
2. **Candidate utility remains poor, demonstrated:** even the nonzero-target
   subset and every training-defined hard slice lose on average.
3. **Directional information is weakly supported, not established:** the fixed
   direction contrasts are small and not uniformly positive by interval.
4. **Causal switchability remains unknown:** an oracle increase is not evidence
   that past-only features can identify its beneficial rows.
5. **Data/representation ambiguity remains unresolved:** annotation jitter,
   interpolated controls, missing intent and the tested representation could
   matter; this loss intervention does not isolate their effects.
6. **Runtime failure is not the explanation:** full budgets, finite outputs,
   exact replay, matched exposure and resume checks pass.

## Scope and Provenance

`fresh_run`: twelve models, replay, fixed analysis and separately registered
post-hoc direction costs. `cached_verified`: twelve unconditional controls and
existing data assets, not twelve additional fresh fits. `not_run`: causal risk
head, independent calibration/confirmation, main benchmark, external transfer,
new deployment or a pretrained visual encoder comparison.

The cohort is 15,430 overlapping stationary-history queries, 29 recordings,
545 scoped agents and four previously explored source sites. Eight observed
annotation steps predict twelve steps, stride 12 raw annotation frames. This
is neither the full mixed-motion benchmark nor the raw-frame t+50 supplement.
Bookstore is excluded from all fits and inference in this experiment; main and
sealed roles are unscored. Two thousand site/recording bootstrap draws are
conditional on this cohort and shared fold training, not independent evidence.

The zero-target cohort has zero baseline error: percentage easy degradation is
undefined, not a <=2% pass. Bounded outputs are not physical-safety proof.
Loaded-target poisoning does not establish sensor-as-of availability of
historically interpolated annotations. Nonzero annotation displacement is not
a human-confirmed motion or intention label. No metric/seconds, true-3D,
foundation or human-gold claim. Stage5C execution and SMC remain off.

## Next Research Decision

Do not start another threshold search on this oracle increase. The next useful
training-side check is whether nonzero changes represent sustained motion or
annotation-scale jitter, and whether causal context contains direction/start
information not captured by the present representation. Keep the original
primary cohort/metric; any magnitude bins are diagnostics, not a new favorable
test. A future learned risk head needs nested upstream producer exclusions,
not an arbitrary split of these OOF labels. More independent scene evidence is
still needed. The broader research goal remains active and unmet.

See [results](results.md), [machine-readable analysis](analysis.json),
[gates](gates.md), [reproduction](reproducibility.md), and
[Chinese operations](operation_zh.md).
