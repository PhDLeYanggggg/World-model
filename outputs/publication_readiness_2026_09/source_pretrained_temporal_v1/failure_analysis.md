# Why The Frozen Visual Candidate Did Not Transfer

## Observed Results

All 36 fixed held fits are negative. The geometry arm nearly reproduces stationary
CV: equal-site gain -0.0704%, window gain -0.1000%. Current-image appearance loses
1.9079% equal-site; eight-frame appearance loses 6.1022%. Sequence-minus-current
is -4.1943 percentage points, conditional four-site interval [-6.2288,-2.6727].
Temporal appearance therefore does not show added forecasting value in this setup.

Appearance does slightly improve fit-cohort error. Current-image site-complement
training gains range from +0.0150% to +0.3531%; sequence gains range from +0.1878%
to +1.4992%. Yet sequence held gains are -3.1527% (coupa), -4.4221% (deathCircle),
-8.4136% (gates) and -9.6619% (hyang). This is an observed fit/transfer gap, not a
proof of its unique causal mechanism. It is not evidence that the training failed
to execute, nor a reason to choose the best held seed.

## Easy Harm And Moving Targets

Post-hoc arithmetic decomposition of the frozen window-weighted scores:

| Arm | Total excess over CV (pp) | Zero-target excess (pp) | Share from zero targets | Nonzero-target gain (%) |
| --- | ---: | ---: | ---: | ---: |
| Geometry | 0.10004 | 0.08094 | 80.91% | -0.01910 |
| Current appearance | 2.09479 | 2.04243 | 97.50% | -0.05237 |
| Eight-frame appearance | 6.50641 | 5.91813 | 90.96% | -0.58828 |

The decomposition uses the same total CV-error denominator, since zero-target
rows contribute zero CV error. It is not the primary equal-site aggregation,
does not remove rows and does not define a new inference rule.
Absolute static-target harms are 0.001644, 0.041487 and 0.120212 annotation pixels.
Percentage easy degradation is undefined here; these are not main safety passes.
The visual models also lose on moving targets on average. Therefore static-label
dominance alone does not explain or repair the failure. The earlier matched
static-gradient removal already worsened actual forecasts substantially.

## What Is Ruled Out And What Is Not

- This is not a hidden all-masked visual run: model-arm behavior is tested and the current/sequence arms consume the frozen appearance embeddings.
- A raw-image versus feature-cache mismatch was not found: three fixed encoder batches (290 images) replay exactly; all query/image rows align.
- Incorrect continuation, unequal sample exposure or a missed fitting budget was not found: all 36 sampler streams match controls, with 10,000 updates and 640,000 draws each.
- Current and past appearance add harmful held-site predictions despite slight training gains. Scene-specific correlations, limited visual resolution, objective/representation mismatch and insufficient transferable start information remain hypotheses, not individually proven causes.
- ResNet18 is an image encoder, not an optical-flow or video-pretrained encoder. This result does not disprove all visual dynamics models.
- Upsampling 32 x 32 crops to the encoder input cannot create additional source detail. This limitation is known; its causal contribution was not isolated by a matched high-resolution experiment.
- Offline annotation interpolation and only four explored physical sites limit interpretation. More overlapping windows or bootstrap draws do not create more independent scenes.

## Why Another Threshold Sweep Is Not The Repair

The equal-site binary candidate/CV oracle gains are 0.0076%, 0.4003% and 1.0563%
for geometry, current and sequence. These future-informed ceilings concern each
fixed arm separately, averaged across seeds. They are not learned performance
and not an oracle over a newly enlarged model portfolio. They indicate limited
useful actions for a gate restricted to these candidates. Do not train or tune a
gate on final test data to manufacture an advantage.

## Next Falsifiable Step

First inspect source-only temporal information loss at the existing image-cache
boundary: frame identity, image repetition, crop resolution, temporal feature
variation and neighborhood coverage. Keep all current rows and roles unchanged.
This can distinguish a recoverable input bottleneck from merely adding another
trajectory head. Reuse the existing native-detail and optical-flow controls,
which already failed on the distinct 11,966-query main-fit population; do not
repeat them or assume that higher resolution alone will fix this source subset.
Any new representation comparison should have fixed matched
controls, unchanged loss/cohort and a pre-fit registration. No new comparison
has been run in this report. A later risk head must exclude its validation site
from every upstream predictor and normalizer, not just its own training rows.

No new deployment, true-3D, physical-time, metric-safety or foundation-model claim.
Stage5C and SMC remain off. The long-term research goal is active and unmet.
