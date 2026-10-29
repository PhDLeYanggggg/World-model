# Easy-Constrained Learning: What Changed and What Failed

## Result

The fixed method experiment completed all 72 fitting comparisons. It repairs a
donor-training objective conflict but does not establish a safe or better policy.
Deployment is unchanged. No independent evaluation or new neural forecaster was
run. The negative conclusion is based on actual policy effects, not on imposing
another risk-MSE gate.

## Hypothesis and Intervention

The previous common blend weight minimized all-risk and easy-risk squared error
together. We hypothesized that this permitted improvements in the larger all-risk
channel at the expense of easy risk. We reconstructed each nested donor objective
with exactly the same neighbors, normalization and original predictions, then
constrained the easy objective not to exceed its original value. This is a fixed
quadratic solution, not a threshold search or new neural training.

Of 62,902 repeated query occurrences, 16,261 have the specified donor-objective
conflict. The constrained solution reduces the blend weight on 17,221 occurrences;
37,569 return weight zero and 2,900 use weight one. The donor easy-loss sum changes
from 50.998271 to 50.889516; the previous blend instead produces 51.250196.
There are zero donor easy-constraint violations. Shared donors mean these are
objective diagnostics, not independent observations or statistical significance.

## Policy Evidence

The unchanged causal rule selects when predicted benefit exceeds harm and both
predicted all/easy harm-reference ratios satisfy 2%. New risk estimates replace
only the original queried subset; every other row keeps its frozen prediction.
Evaluation includes the complete eligible fitting population, not only known
labels or selected queries. Unknown outcomes retain their registered bounds.

| Comparison | Full-policy completion interval (pp) | Count-matched interval (pp) |
|---|---:|---:|
| Primary minus original | [-2.879768, -1.570854] | [-0.692110, -0.454223] |
| Primary minus previous blend | [-0.466932, 2.183365] | [0.097505, 0.165584] |

These are equal-view averages of deterministic partial-label bounds, normalized
by each view's full known reference mass. They are not bootstrap confidence
intervals, independent test effects, or relative ADE/FDE improvements.

The small positive count-matched effect versus the previous repair is a useful
local signal. It neither beats the original policy nor establishes safety. Even
the upper bound versus original is negative. The full comparison with the
previous repair spans zero: do not call that a proven true-effect regression.

The primary makes 51,693 interventions, including 1,139 with missing outcome
labels. Only 5/72 view budgets and 29/444 recording-view budgets are supported.
The 444 entries repeat recordings across contexts/head seeds, not 444 independent
scenes. All required entries remain present.

- 33 views exceed the all-risk budget on known labels alone.
- 22 views exceed the easy-risk budget on known labels alone.
- Another 30 all-risk and 37 easy-risk view failures arise from the unknown-label
  completion bound, despite a defined known-label ratio within budget.
- Seven easy ratios are undefined. These are not safe zero risks.

The categories overlap. Lack of labels limits identification, but cannot explain
away the observed known-label failures. Improving annotation alone is insufficient.

The two empty count-matched previous-blend controls remain in utility accounting;
their risk is explicitly not estimable and is not a full-policy safety gate.
This follows the user-confirmed prospective protocol, without changing 2% or
rewriting any historical failed result.

## Failure Taxonomy

1. **Objective conflict is real but not sufficient.** The new constraint fixes
   the measured donor conflict without recovering complete-policy safety.
2. **Squared-error preservation is not decision preservation.** A lower donor
   risk error does not ensure correct ranking near the intervention boundary.
   Primary easy query MSE still exceeds original, and full/count-matched utility
   versus original is negative under the stated completion bounds.
3. **Shrinking risk is not locating harm.** Primary easy MSE still loses to a
   recording-mean weight control. This argues against attributing the small
   improvements to reliable query-specific risk discrimination.
4. **Both prediction and label support matter.** Known-label violations persist;
   unknown selected outcomes additionally prevent budget support. Neither failure
   should be concealed by dropping recordings or unknown rows.
5. **Evidence remains exploratory.** Upstream heads/preprocessing are exposed to
   TRAIN. Excluding query recordings from donor fitting does not make the whole
   pipeline out-of-fold. Three head seeds are not three new forecaster seeds.

## Next Decisive Experiment

Stop treating another risk-MSE reduction as the route to success. Use the existing
causal features and frozen forecasts to compare decision-aware gain/harm learning
and scene-level constrained intervention against the original and an independent
agent selector. Hold the 2% budget, data roles and comparators fixed. Primary
readouts should be actual utility at matched intervention counts and complete
policy safety, with risk error only explaining the decisions.

The known-label failures justify changing the learning objective before spending
independent scenes on model selection. Missing-label effects should be reported
separately; do not use future knownness as an inference filter. This next method
experiment is not run in this package. No new user decision is needed for the
already authorized train-only implementation and experiment.

## Provenance and Limits

- `fresh_run`: pilot, 72 donor-constrained fits and full/matched fitting readout.
- `cached_verified`: prior control predictions, reference/neighbor identities,
  registered source packets and separate aggregate arithmetic.
- `not_run`: new neural training, independent selection/calibration/confirmation,
  deployment promotion or confirmatory uncertainty estimation.

obs8/pred12, stride12 raw frames, pixel/local detector-silver. No metric, seconds,
human-gold, physical-safety, true-3D or foundation claim. Stage5C and SMC are off.
The paper goal is active and not achieved. This is a completed falsification and
mechanism diagnosis, not evidence that M3W is submission-ready.

[All six arms and slices](readout.md). [Registered protocol](protocol.md).
