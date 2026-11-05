# Failure And Mechanism Analysis

The prespecified pilot expansion conditions fail. This loss variant is not promoted to a full study or deployment.

## What Was Tested

Earlier signed-risk penalties on constrained cost moments, log-harm losses,
event heads, and proposal-only supervision were not silently repeated or
reclassified as successes. This is a narrower unbounded signed-output test.
Mean versus quantile isolates the loss within that new architecture; either
versus the old critic changes both parameterization and supervision.

## Fitting Versus Held-Source Tail

The following pooled known-row excess sums are descriptive, in local pixel
units and repeated context views. Columns are all/easy; unknown bounds stay
in the policy evaluation but are not misrepresented as realized outcomes.

| Scope | Head | Set | Known rows | Predicted excess sum | Actual excess sum | Fraction actual > predicted |
|---|---|---|---:|---|---|---|
| fit | mean | proposals | 48061 | 393260.344388, 193777.584695 | 235500.783506, 45001.300410 | 0.401469, 0.384345 |
| fit | mean | admitted | 11813 | -48393.542387, -47537.178190 | 5212.361480, 2000.147472 | 0.898671, 0.993651 |
| fit | quantile90 | proposals | 48061 | 1055950.506244, 546684.754502 | 235500.783506, 45001.300410 | 0.069225, 0.060506 |
| fit | quantile90 | admitted | 610 | -942.726957, -312.737079 | -910.607001, 7.776868 | 0.691803, 0.942623 |
| held_SOURCE | mean | proposals | 78665 | 421363.823969, 309281.105942 | 63859.061813, 37718.028649 | 0.459734, 0.531685 |
| held_SOURCE | mean | admitted | 34888 | -238686.556342, -263057.237617 | 17382.627249, 13115.115846 | 0.938776, 0.997965 |
| held_SOURCE | quantile90 | proposals | 78665 | 1740552.318572, 1112290.277216 | 63859.061813, 37718.028649 | 0.066345, 0.122532 |
| held_SOURCE | quantile90 | admitted | 2543 | -9091.600691, -8092.235079 | 3555.517733, 2441.239092 | 0.904837, 0.988596 |

## Reading The Pilot

Mean minus old conditioned known utility: -1.240643pp; quantile minus old: -5.180360pp.
The quantile switch rate is 1.519389%, versus 31.794134% for the old conditioned veto.
These are the identical seven pilot SOURCE units, not the previous full
23-source experiment. Neither point estimates nor this one seed establish
independent significance. See every source/context contrast in analysis.

The fit-versus-held table matters: if the admitted fitting subset already
has underestimated excess, the failure cannot be attributed solely to
unseen-scene shift. Marginal tail coverage and conditional admitted-set
error are different. Underprediction of a negative budget excess does
not itself mean harm; actual positive excess and full record gates must
still be inspected. Conversely, low marginal underprediction frequency
does not certify selected mean harm. See the pre-readout method note.

## Limits And Next Decision

Training fit, held prediction error, and policy safety are separate tests.
A lower fraction of underpredictions is not by itself useful intervention:
excessive abstention can lose benefit while still leaving selected harm or
missing-support failures. Quantile .9 does not bound a selected-set mean.
The observed signed excess also depends on rare easy-reference support.
Sparse support is retained as NA, not turned into a certificate.

This pilot is limited to the preselected contexts and one seed. A negative
pilot does not disprove every tail learner or demonstrate irreducible
future ambiguity. It does rule out promoting this particular repair from
its fitting loss alone. Further work must distinguish lost beneficial
actions from retained harmful actions using fixed causal slices before
another model/threshold sweep. More computation is not evidence by itself.

Original2%harm, support, easy-net and empty-control rules are unchanged.
No new H, independent confirmation, physical-safety, metric, seconds,
true3D or foundation claim. Stage5C/SMC remain off.
