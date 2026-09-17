# What the Real Deferral Control Changes

## Completed Evidence

This is a fresh real-data comparator fit and development evaluation, not another
synthetic smoke test. Twenty-four heads completed 1,000 updates: both frozen v7
predictors, seeds 17/29/43, linear and width-64 GELU heads, cost bounds 1/10.
The original forecasting checkpoints were verified and reused. Every head uses
the identical 11,966 OOF rows and 306 causal features of the existing gain/harm
heads. All floor/candidate errors match the completed parent exports exactly
before cached ordinary controls enter the comparison.

The complete resumed invocation took 263.50 seconds; head-fit timers sum to
10.89 seconds, excluding OOF extraction and evaluation. These are small routing
heads, not a shortened forecast-training schedule. The first head had completed
before the resumability smoke check. CPU4/interop1/workers0; all 12 recording
evaluations freshly scored; original ordinary controls are cached_verified.
Final focused verification: 50 tests passed in 17.11s. The full legacy suite was
not rerun; these checks are engineering evidence, not a research gate.

## Results

- CV-skip: all 12 deferrers worsen primary normalized ADE, -0.01035% to -0.70739%.
  Easy degradation is 162.01% to 7,868.83%.
- Motion-bounded: all 12 have small positive average gains, +0.01997% to +0.08445%.
  None passes easy preservation: degradation 14.40--96.96%, versus the 2% ceiling.
- Bounded family mean gains are +0.04954% to +0.07337%, depending on the fixed
  setting. This is not a selected winner or a significance test.
- Easy CV ADE is 0.00323052 past-normalized units. Bounded deferral raises it to
  0.00369566--0.00636283. The small denominator explains large percentages;
  reporting absolute errors does not remove the failed easy gate.
- Bounded hard gains are only +0.02050% to +0.09258%, not a 10% improvement.
  Intervention rates are 10.49--86.86% of all past-supported queries.
- Every head reduces first-to-last-50 minibatch surrogate loss; this is not
  convergence proof or safe forecasting improvement.

All 24 settings fail positive gain plus easy<=2%. One exposed physical development
site, not three independent sites, remains. No physical-scene CI is claimed.

## Interpretation

The real comparator closes a missing control. Keeping continuous costs rather
than an oracle class does not automatically protect easy cases. However, M3W's
guarded controls impose budgets absent from deferral. Their lower easy harm
cannot be attributed solely to a better learning objective. The 480 paired
contrasts reuse development windows; they are not 480 independent experiments.

Around 1.83--4.73% of OOF rows have both costs above the cap, depending on
family/bound. Their bounded labels tie even when raw costs differ. Clipping loses
some information, but this comparison does not isolate clipping as the cause.
Neither cap was tuned or selected using development results.

A routing head cannot exceed its candidate/floor oracle. The v7 oracle ceiling
is below 0.338% under the unchanged primary metric. Routing cannot create missing
motion information. The bound prevents drift but also forbids predicting movement
from an exactly stationary observed path. The prior fit-only diagnostic places
73.3% of pooled fit CV error on those stationary-start cases. This is a concrete
capacity limit, not a reason to discard cases or retrospectively switch metrics.

## Research Decision

Do not add another threshold grid to these candidates or promote a deferrer as
deployable. Joint routing still lacks support: the completed matched-count
experiment finds no gain over independent decisions.

Next, prospectively audit fit-only stationary-to-moving behavior versus annotation
and scale artifacts. Test whether past neighbors or scene cues predict starts
beyond a fit-only start prior. Retain all targets and frozen v7 as a reference.
Only if held-fold signal exists should a registered start-aware predictor follow.
Any new primary metric or scientific split needs a separate decision.

These comparisons and capacity diagnostics can support a paper's methods and
limitations, not its main superiority claim. Independent eligible sites,
consistent geometry/time provenance and a positive contribution beyond existing
regression deferral are still missing. No Stage5C, SMC, metric/seconds, independent
confirmation, new deployment or submission-ready claim.

See [all settings](results.md), [metrics](metrics.json) and the [fixed decision](../deferral_v7_decision.md).
The parent protocol and its original source hashes were not modified.
