# Baseline-Relative Repair: Complete Primary Study, Limited Gain

Date: 2026-09-17. Fresh real training and development evaluation, not independent
confirmation. Both arms and all seeds 17/29/43 complete the registered budget:
24 forecasters at 10,000 updates, six neural cost heads at 1,000 updates, six
OOF ridge controls and 120 arm comparisons. Neural fitting totals 4,693.94 seconds
across both families; this excludes OOF extraction and evaluation and is not
the wall-clock duration of the whole study. Runtime is arm64 CPU4, interop1,
workers0, with checkpoints and heartbeat. No budget was shortened.

## The Paired Question

Both versions add a neural residual to the declared causal CV trajectory,
starting with an exactly zero residual. The second version only changes the
output map: its correction radius is determined by observed ego path length,
causal CV extent and requested time. The matched comparison isolates that
amplitude restriction. Comparing with v6 additionally changes skip connection
and output initialization; it is not a bound-only comparison.

| Seed | v6 absolute model gain vs CV | CV-skip gain | Motion-bounded gain | Bounded easy degradation without selection |
| --- | ---: | ---: | ---: | ---: |
| 17 | -5.73556% | -0.562987% | +0.075900% | 62.5809% |
| 29 | -7.68637% | -0.679793% | +0.056063% | 398.5673% |
| 43 | -6.69954% | -0.552938% | +0.052557% | 308.0666% |

The past-normalized ADE gain averages -6.7072%, -0.5986% and +0.06151%,
respectively. The bounded model's training-seed SD is 0.01259 percentage points,
not a scene confidence interval. Its easy ADE rises from CV's 0.00323052 to
0.00525221 / 0.01610633 / 0.01318268. A small positive overall gain therefore
does not establish an unguarded deployable model.

## Selection Under the Unchanged Rule

The original development rule requires positive primary gain, <=2% easy
degradation and the fixed predicted-budget guards. It selects the following
policies without a new threshold search:

| Seed | CV-skip selected gain | Bounded selected head/policy | Bounded selected gain | Hard gain | Easy degradation | All-past-supported switch rate |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 17 | +0.00048886% | neural cost, moderate, independent | +0.00249499% | +0.00239080% | 1.18890% | 3.46790% |
| 29 | +0.00033277% | ridge, conservative, independent | +0.00771028% | +0.00809655% | 1.88251% | 3.78822% |
| 43 | 0%, CV floor | neural cost, conservative, independent | +0.00051883% | +0.00040938% | 0.25144% | 0.59828% |

These gains are deliberately not rounded into a success headline. The selector
family differs by seed, and selection and reporting use the same development
site. They are not the result of one independently tested deployment policy.
Candidate/CV label-oracle headroom is only 0.162%, 0.337% and 0.249% for the
bounded arm. Even perfect routing of these fixed candidates cannot yield a 5%
gain on this primary development metric. More threshold search is not the fix.

## What the Repair Does and Does Not Explain

The fixed-result error decomposition shows exactly zero extra bounded-model
error on the 3,082 complete queries at the existing numerical scale floor. This
is consistent with the construction, not a learned ability to predict starts.
All remaining improvement and harm occur above that floor. The CV-skip version
still causes positive net harm at the floor and modest net improvement above it.
The observations support drift reduction under this controlled comparison; they
do not establish a general safety theorem or a new joint-decision contribution.

The separate fit-only geometric diagnostic retains every future label. Its
365 zero-budget rows account for 73.253% of pooled fit CV error. A label-aware
per-step correction-ball oracle has at most 4.089% average fit-window headroom.
That is an optimistic capacity limit for this bound on training data, not a
learned score or the equal-scene development primary metric. Freezing a stationary
prediction protects some easy cases while necessarily missing later starts.

Native-coordinate bounded gains over the development-best causal alternative
are +0.559% to +0.973% on Students01 but -2.106% to -4.919% on Students03.
Positive CV-only native gains do not mean a consistent win against the stronger
causal alternative. Raw coordinate errors are not pooled across recordings.

## Research Decision

This completes a falsifiable repair, not a paper-ready method. Keep the full
negative/easy evidence and do not promote a new deployment. Primary support is
28,324 complete ADE queries, 28,335 endpoint-valid queries and 37,775 past-supported
queries. Two recordings still represent one historically exposed physical site.
No independent scene interval, final test, risk certificate, metric/seconds,
true-3D, foundation, Stage5C or SMC claim follows.

The shortest next research path is to explain useful candidate disagreement and
the failure to improve mixed predictions at matched count, before adding another
architecture. The stationary-start constraint and normalization sensitivity must
be addressed prospectively with a declared task/metric, not by excluding hard
rows or replacing the failed metric after seeing results. Independent, source-
eligible sites and a compatible public-benchmark protocol remain essential for
a submission-quality claim. Supplementary results are recorded separately; they
cannot choose a new policy or override the primary evidence.

Both frozen supplements are now complete. CV-skip matched-count identities are
identical; bounded matched routing has nine zero differences and three tiny
ADE changes favoring independent selection. Uncontrolled bounded raw50 gains
are positive for two seeds and negative for one. The original errors and
ordinary decisions replay exactly. These results do not establish joint lift.

## Evidence

- [Paired results and native causal context](residual_parameterization_comparison.md)
- [Fit-only bound capacity](fit_bound_headroom.md)
- [All CV-skip controls](../8to12_residual_skip_v7/results.md)
- [All bounded controls](../8to12_motion_bounded_v7/results.md)
- [CV-skip error decomposition](../8to12_residual_skip_v7/error_scale.md)
- [Bounded error decomposition](../8to12_motion_bounded_v7/error_scale.md)
- [Complete supplementary conclusions](supplement_conclusions.md)
