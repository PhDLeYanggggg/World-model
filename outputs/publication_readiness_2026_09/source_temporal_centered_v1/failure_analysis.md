# Temporal Centering: Less Harm, Still No Forecasting Benefit

## What Was Tested

The past-only input audit found real temporal variation, not empty or repeated
frame sequences. All 123,440 historical keys align; all 15,430 windows have eight
supported observations. However, only 2.5715% of frozen embedding energy is
within-window variation on average. This motivated a fixed comparison of
mean-centered appearance and mean-centered appearance with per-window RMS
normalization. Shared appearance is not necessarily background, and pixel changes
are not verified motion intention.

Twenty-four fresh neural heads completed 240,000 updates. The previous 36 heads
are hash-verified matched controls, not new training. Cohort, seeded initial
parameters, sampling, loss, geometry, coverage and fitting budget are unchanged.
There was no held-out arm, seed, checkpoint, threshold or sample selection.

## Actual Results

| Arm | Equal-site ADE gain vs stationary CV | Conditional four-site 95% interval | Absolute zero-target harm (annotation pixels) |
| --- | ---: | --- | ---: |
| Original sequence, cached control | -6.1022% | [-9.1708%, -3.8073%] | 0.120212 |
| Centered sequence, fresh | -0.7622% | [-1.7428%, -0.1231%] | 0.020320 |
| Centered, RMS-normalized sequence, fresh | -1.7564% | [-3.4247%, -0.5537%] | 0.041864 |

Centering improves the failing sequence control by 5.3401 percentage points,
conditional interval [3.6666, 7.5778]. That is damage reduction, not a positive
forecasting contribution: all 24 new held fits remain negative. Both new arms
also lose to the geometry-only control. Centering-minus-geometry is -0.6918pp,
interval [-1.5819, -0.1041]; normalized-centering-minus-geometry is -1.6860pp,
interval [-3.2638, -0.5347]. All five registered contrasts are reported.

## Failure Taxonomy

| Candidate cause | Evidence | Supported interpretation |
| --- | --- | --- |
| Empty images, repeated frames or row mismatch | Key audit, encoder replay and query alignment pass | Not found; these checks do not establish useful predictive content. |
| Common appearance or its scale dominates the readout | Centering reduces excluded-site error substantially | A harmful input/readout component was reduced. Centering changes both the shared signal and input magnitude, so background causality is not isolated. |
| Weak temporal signal merely needs amplification | RMS-normalized arm is 0.9942pp worse than centered, interval [-1.7136, -0.4306] | Fixed amplification does not repair transfer. Amplified nuisance variation is a hypothesis, not a measured unique cause. |
| Static-label domination alone | Nonzero-target gains are also negative: -0.0614% and -0.1679% | Easy-case harm is substantial, but moving-target prediction is not repaired either. Earlier static-gradient removal also failed. |
| Training interruption or unequal exposure | All 24 heads reach 10,000 updates; 24 sampling streams match controls | Not a shortened run or hidden exposure advantage. |
| Scene-specific fitting | Hyang-complement training gains are +0.2371%/+0.5470%; corresponding held gains are -2.3182%/-4.4044% | A fitting/transfer gap remains. Neither a unique causal mechanism nor unseen-domain success is established. |
| Fallback threshold is the only remaining issue | Per-arm binary-oracle gains are just 0.1630%/0.3368% | The fixed candidates offer little action headroom. Future-informed oracle choice is not a deployable policy. |

The hard-subset gains are tiny: approximately -0.001% to +0.018% for centered and
-0.003% to +0.062% for normalized-centered across sites. These are not a 10% hard
improvement or a reason to ignore negative overall predictions. Hyang is worst
for both arms; it remains included and its three seeds are all reported.

## Safety And Statistical Limits

For zero-target rows, stationary CV has exactly zero error. Percentage degradation
is undefined, not a 2% preservation pass. Absolute harm is reported instead.
Native pixel averages are descriptive within this source diagnostic, not calibrated
physical errors or cross-dataset metric results.

The 2,000 bootstrap draws resample four physical sites, not overlapping windows.
They remain conditional on explored sites and shared fitting folds. Three seeds
do not turn these sites into an independent confirmation set. Later annotation
controls can affect supplied histories: this is the approved offline annotation
protocol, not strict sensor-as-of perception.

## Next Work And Stop Rule

Do not promote either arm, sweep a new switch threshold on these held outcomes,
or describe full fallback as learned success. Do not repeat the completed native
detail, optical-flow, static-loss or pretrained-centering controls unchanged.

Before another representation fit, assess independent event support using only
the existing development/training roles: distinct agent departures and their
recording distribution, the available pre-query visual/neighbor history around
them, and whether repeated windows are providing new information or just repeated
labels. Reuse the previous row-quality and duplication audits rather than count
windows as independent events. Any support bins are descriptive, not permission
to discard difficult or static test rows.

The next candidate must add identifiable past information or independently useful
state-change support. A fixed comparison should test its actual trajectory gain,
not merely probability ranking, latent variance or a larger oracle. If independent
event support is inadequate, prioritize admissible additional data over a larger
head on the same frozen representation. This support investigation is not run in
the present report, and no new fitting recipe has been selected from these results.

Main/outer evaluation, new policy, external testing and independent confirmation
are not run. No new model is deployed. No metric, seconds, true-3D or foundation
claim is justified. Stage5C and SMC remain off; the research goal is unmet.
