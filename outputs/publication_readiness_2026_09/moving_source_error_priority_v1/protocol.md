# Source-Held Error Discovery

## Prospective Question

Does discovering false-safe errors on an unseen fitting source help more than
discovering errors on a source already seen by an equally sized discovery
model? The preceding fixed warm-error experiment reduced fitting error but
lost held-source utility. Its failure remains unchanged.

## Fixed Roles And Comparison

Use the same three seed17 development contexts as the preceding study. A=part1
remains the frozen foreign proposer, B=part0 remains fitting, V=part2 remains
already exposed SOURCE validation. B contains two/three/three source groups.
This is not an independent test or a three-seed study.

Inside each B, fit one new 32,130-parameter kinematic risk head per omitted
source, from random initialization. All common preprocessing, target RMS and
64-field normalization are fitted on B minus that source. No full-B warm
weights or full-B normalization enter these eight discovery heads. Producer
forecasts and producer-only easy/hard definitions remain frozen.

For source s, the held-error arm uses the teacher omitting s. The in-source
control uses the teacher omitting the next source in sorted circular order.
Thus each row uses exactly one teacher in each arm, of equal architecture,
source count and update budget. Remaining row counts and difficulty differ;
they are reported rather than claimed perfectly matched. No outcome-dependent
teacher choice or ensemble. Both sets of teachers can be replayed exactly.

Observed legal channels with predicted signed risk <=0 and realized signed
budget excess >0 receive weight10, using the unchanged earlier objective.
Unknown outcomes retain their bound-aware anchor and never receive error
priority. Discovered errors are supervision only, never inference inputs.

Fit two student heads per context from the SAME frozen full-B kinematic model,
with identical reset AdamW, sampling, preprocessing, architecture and 2,000
updates. Only error-mask provenance differs. The reused trainer field named
`warm_prediction_hash` stores the discovery prediction hash here; the true
student warm checkpoint and teacher mapping are separately bound in identity.
Random discovery initialization is explicitly step0, not a completed warm fit.

## Sequence And Decision

Register and commit before real fitting. Verify exact resume with a real
50+50 versus100 discovery pilot. Train all eight teachers, freeze their masks,
then all six students before any V readout. Independently repeat the first
full teacher and student, then repeat all predictions/actions/metrics.
Use the original zero signed-excess admission and original support checks;
no threshold, weight or favorable-context search.

Expansion requires positive missing-outcome completion lower utility contrast
for held-error versus in-source error in EVERY context, nonzero intervention,
and no increase in known all/easy violations or undefined selected/full easy
views in any context. Aggregate lower contrast versus the original affine
readout must also be positive. This is a development-screen condition, NOT a
safety certificate. Deployment requires the unchanged full original rules.

## Evidence Limits

Native obs8/pred12, image-pixel coordinates; no seconds or metric claims.
Completion bounds are not confidence intervals. Original 24-group primary
remains not estimable where reference support is absent. Empty matched
controls remain NA. No identity certificate. Reserved20 Moving groups, DUT,
DroneCrowd and H stay closed. No new trajectory producer/JEPA training.
Stage5C/SMC remain disabled; no deployment or submission-ready claim.

Runtime: native arm64 Torch CPU4/interop1, single process, checkpoints and
heartbeat every100steps, 12h guard, 5GiB disk reserve, 512MiB private cap.
