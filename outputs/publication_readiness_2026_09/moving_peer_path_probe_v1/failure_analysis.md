# Directional Peer Paths: Information Without Reliable Decisions

## Question And Result

The previous 24-feature neighbor probe and the matched neural participation
factorial did not produce reliable decision gains. I tested a distinct missing-
information hypothesis: proximity summaries may discard the direction and time
profile of a candidate's relation to nearby agents. Sixty past-derived path
features add heading-aligned nearest-peer vectors for the floor and candidate,
plus their radial-distance difference, at all twelve prediction steps. The peer
path is a causal constant-velocity forecast, not observed future motion.

The fixed primary nonlinear probe passes **1/3** internal context screens,
below the registered requirement of at least two. It does not justify expanding
this feature branch into a new neural training campaign. This is a completed
negative follow-up decision with a localized information signal, not a proof
that interaction modeling is impossible.

## What Changed

- Summary control: history64, rollout32 and the old neighbor24 features.
- Aligned treatment: the same120 features plus the new60 directional features.
- Association control: the same180 dimensions, but the new60 features come
  from another row in the complete current query, assigned before label filtering.
- Fixed ExtraTrees is primary; ridge is secondary. Each leaves out one fitting
  source, and learns normalization and target scale on the remaining sources.
- No loss/gate/threshold search, new forecaster, new neural dynamics, nested-C
  evaluation or reserved-data readout occurred.

The mismatch keeps the new feature block's query-level multiset, not its
association with the candidate. It also changes correlations with the original
120 fields, so it is a limited association control rather than an exact causal
intervention. There are no singleton queries in these inputs. All rows have
at least one aligned observed neighbor under the existing mask definition.

## Three Different Failure Patterns

| Context | Primary MSE reduction | Joint-cost reduction | Outcome |
|---|---:|---:|---|
| 0 | -1.262274% | -0.000449% | Worse MSE than both summary and mismatch |
| 1 | +0.996725% | +0.189098% | More useful admissions, but also more harmful ones |
| 2 | +5.000471% | 0.000000% | Better continuous prediction, no decision changes |

**Context0: no demonstrated incremental information.** Both source folds lose
MSE, and the aligned block is worse than the mismatched block. All three arms
admit one known candidate; equal counts do not imply identical row choices.

**Context1: the tradeoff is not safe.** Known admissions rise from10 to61;
false-safe admissions rise from1 to10. False rejections fall2239 to2197, so a
small aggregate cost improvement coexists with increased harmful-admission
cost. Source-macro false-safe cost rises from4.2984456e-7 to3.6856046e-6.
The +757.43% relative increase has a small denominator; both absolute values
and counts are reported. The exact MSE improvement is below1%, not a rounded
pass, and harmful cost independently fails the screen. Source065 has only44
known proposals across19queries, limiting its precision. No source is removed.

**Context2: regression improvement is not utility improvement.** All arms
reject all38,579 known proposed candidates, including24,216 with nonpositive
targets on both original channels. MSE improves5.0005% over the summary and
6.5807% over the mismatched control. Most source-level MSE signal is in059:
+7.159%, compared with+0.559% in060 and+0.218% in073. The joint cost is exactly
unchanged. Zero harmful admissions is not evidence of useful switching or a
full-policy safety pass, especially with unknown outcomes retained as unknown.

The secondary ridge also fails all three screens. In context2 it improves MSE
1.7234% but worsens joint cost3.1563%. It cannot replace the registered primary.

## Supported Diagnosis Versus Hypothesis

Established: a single continuous-error objective and the downstream sign decision
can disagree. The new information is not uniformly useful across sources, and
the most favorable MSE result does not change decisions. Missing neighbor
availability is not the explanation in these particular rows. More columns
or more model fits alone are not a repair.

Not established: whether rejection comes primarily from tail-severity averaging,
tree regularization, omitted causal state, or source shift. These probes cannot
identify that cause, nor establish the absence of useful interaction dynamics.
The source forecasters already exposed these development sources; excluding a
source only from the probe fit does not create independent confirmation.

## Next Model Question

Do not expand the directional-path neural branch on this evidence, drop source0,
promote the secondary model, or tune admission thresholds. First audit the
training-source signed-target distributions and the fitted prediction signs:
does the cost-weighted tail erase a learnable useful/harmful separation, or is
the conditional information itself missing? Compare that diagnosis against the
existing negative paired-decision and cost-head experiments before proposing a
new loss. A probability-times-severity decomposition would be a new hypothesis,
not a demonstrated solution or a reason to change the scientific risk budget.

## Evidence Boundaries

Fresh:48 probe-fit executions, their full deterministic refits, causal features
and report calculations. Eight fits intentionally reproduce old ridge controls.
Cached and verified: source inputs, producer forecasts, split/schema hashes and
the old controls. Not run: new neural training, additional nested-C/H, reserved
sets, independent confirmation or deployment. Overlapping rows/queries and
three fitting contexts are not three independent seeds. No inferential CI is
claimed for this exploratory information test.

Original2% all/easy harm, signed easy-net, reference/support and missing-label
rules stay unchanged. No identity certificate. Empty support remains NA. The
old17-view safety failure and original24-group primary NA remain. Moving20,
DUT, DroneCrowd and H remain closed. Native obs8/pred12 image-coordinate silver
data are not metric, seconds-level, true3D or foundation evidence. Stage5C and
SMC are off. This work does not make the paper submission-ready.
