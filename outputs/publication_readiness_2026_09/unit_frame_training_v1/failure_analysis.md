# Failure Taxonomy: Conditioning Does Not Establish Forecast Skill

## Confirmed Observations

1. **Numerical unit dependence:** v1 rollout features inherited a native-speed
   cutoff. A tiny curved-history rescaling test fails before v2 and passes
   after recomputing rollout features internally. This is a real input repair,
   not proof that a neural predictor is better.
2. **Optimization under-response:** on fixed training probes the frozen legacy
   model's source static-to-moving per-row gradient norm is0.003087, versus
   1.449280 for other motion. Actual autograd supports the measured imbalance,
   not an attribution that it alone causes forecast failure.
3. **Decoder amplification:** all189 logged source batches and358/360 logged
   main batches exceed the gradient clip norm in the unit-decoder/primary-loss
   arm. These are sparse trace counts, not counts of all training batches.
4. **Objective mismatch:** the internal-loss arm has no clipping in its logged
   batches, yet aggregate primary gain is -185.77715%. Small internal loss and
   balanced gradients are not aligned with primary ADE or easy preservation.
5. **Cross-site instability:** inputs-only training gain is positive in every
   fit; Hotel held gain is positive, ETH/Zara are negative. Overall gain remains
   -0.68730%. A generic unit repair does not resolve scene-domain differences.
6. **Unrecovered start information:** inputs-only static-start gain is-0.00509%,
   unit-decoder -0.26601%, internal-loss -31.70294%. Improved stopping is not
   transferable start/direction prediction.
7. **Unsafe drift:** all27 new fits fail easy preservation. Internal-loss
   static-stay mean harm is77.71223 normalized ADE versus an exactly zero CV
   reference. This is not only a tiny-denominator reporting artifact.
8. **Limited candidate ceiling:** even a future-label oracle over the expanded
   five-choice pool gives2.63722% aggregate gain. It cannot establish that a
   trainable causal selector will reach that ceiling or a larger headline gain.

## Mechanism Interpretation and Limits

For fixed positive past radius R and primary ADE e, the internal objective is
log(1+e/R), with derivative1/(R+e) with respect to primary error. A large context
radius therefore permits appreciable primary error for a small internal loss.
Multiplying predictions by R also expands small internal drift. This algebra
and the observed failures support an objective/decoder mismatch explanation.
They do not identify the optimal new loss or demonstrate a safe risk controller.

The scale uses the farthest selected observed ego/neighbor position; it is not
a verified physical mobility scale. Future motion remains partially observable
from eight annotated positions. SDD and main temporal spacing differ, and image
features are intentionally absent from this mechanism comparison. The experiment
cannot falsify all multimodal models, prove scene images are irrelevant, or
distinguish every data/architecture cause. Main target agents and source queries
are pedestrians, so mixed *target-agent* type is not a demonstrated cause here.

No-anchor guarding changes the old aggregate by only0.00496percentage points.
It is not the main explanation. All checkpoints and predictions are finite;
this was not an OpenMP, SHM, MPS, NaN, resume or training-crash failure.

## Retained and Rejected

Retain the past-only v2 feature reconstruction, its regression tests and exact
resume machinery as engineering assets. Retain the stopping-slice signal as an
exploratory observation, not an endpoint chosen after seeing results.

Reject deployment of all new models and reject expanding the unqualified
radius-decoder/internal-loss recipe. Do not reopen sealed roles or search their
thresholds. The next falsifiable action is training-role benefit/harm and support
analysis with explicit decoder sensitivity, followed only by a registered
matched experiment justified by that analysis. Independent-scene confirmation
and the proposed joint-intervention contribution remain unresolved.
