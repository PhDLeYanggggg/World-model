# Protected Motion: Completed, Neural Trajectory Advantage Not Established

The registered comparison is complete. Forty-five damping-candidate heads were
fitted afresh: 18 ridge and 27 real Torch heads, 54,000 neural updates. Forty-five
neural-candidate utility/risk heads and the upstream trajectory predictors were
cached_verified, not retrained. Both decision banks were freshly executed.
All 48 policy views are reported; no seed, threshold or winner is promoted.

## Primary Result

Giving fixed damping 0.97 the same candidate-specific gain/harm learning, source
folds, fitting draws and predicted-risk rules removes the apparent case for the
neural trajectory candidate in this comparison. Across all 24 matched pointwise
contrasts, neural gain over protected damping is negative, from -1.8329% to
-0.2226%; every conditional locality-bootstrap interval is below zero. The same
holds for all 24 hard-subset contrasts and all 24 joint-pilot contrasts.

This is a comparison of complete protected-candidate pipelines. It does not
isolate forecast quality from the learnability of candidate-specific utility
and risk, and it does not show that neural forecasting is universally inferior.
The intervals share fitted producers and source localities; they are not 24
independent confirmations or multiplicity-adjusted population guarantees.

The following easy-event neural-risk views continue the previous report's
comparison; they are not a post-readout model selection. All other views remain
in [results.md](results.md) and [summary_metrics.json](summary_metrics.json).

| Seed, no support guard | Neural trajectory ADE gain vs CV | Protected damping ADE gain vs CV, conditional 95% CI | Direct neural gain vs protected damping, CI | Damping intervention rate |
|---|---:|---|---|---:|
| 17 | 0.2382% | 1.8108% [0.9706%, 2.6765%] | -1.6240% [-2.4745%, -0.7891%] | 21.0287% |
| 29 | 0.1663% | 1.9388% [1.0465%, 2.8589%] | -1.8329% [-2.7402%, -0.9433%] | 21.9661% |
| 43 | 0.4310% | 1.8975% [1.0439%, 2.7661%] | -1.5127% [-2.2122%, -0.8246%] | 18.7746% |

The damping easy-event neural-risk views improve, rather than degrade, easy
ADE in every observed locality and harm none of the four observed zero-CV cases.
With the source-zero-support guard, damping gains are 1.7827%, 1.8991%, 1.8661%
versus 0.2141%, 0.1661%, 0.4275% for the neural trajectory candidate. Damping
all-event ridge also achieves 1.7330--1.7698% with observed pointwise protection.
Neural risk learning and neural trajectory forecasting are different claims.

Equal-locality percentage gains are not pooled pixel ratios, and the direct
neural-versus-damping gain is not obtained by subtracting two CV-relative
percentages. Equal predicted-risk budgets do not impose equal realized risk
or intervention counts between candidate pipelines.

## Safety and Joint Controls

Twenty-two of 24 neural-candidate and 21 of 24 damping-candidate full-pointwise
views satisfy the observed worst-locality easy <=2% and zero-CV added harm 0
checks. The failures remain: neural easy-ridge unguarded harms one zero-CV case
in seeds 17 and 29; damping easy-ridge unguarded has worst-locality easy
degradation of 2.8904%, 2.8579%, 3.3194%. Mean easy gains do not erase these tails.

The joint pilot uses 1,152 queries and 6,116 targets, not the full cohort. All
24 neural-versus-damping joint intervals are negative, with gains from -2.8381%
to -1.1079%. Within-candidate exact-count contrasts do not establish a stable
joint advantage: no defined joint-versus-independent or joint-versus-unary
interval is strictly positive. Fifteen neural and twelve damping exact-count
contrasts are undefined for lack of supported nonzero-match localities; they
are not converted to zero or recomputed after dropping a locality.

Only 13/24 neural and 20/24 damping joint views meet observed easy/zero checks.
There are no zero-CV cases in this pilot, so these checks cannot validate zero
event safety. All four full-cohort zero-CV cases come from one locality, each
with only two of twelve future labels and no endpoint. Their outer fitting
fold has no zero-event support. The source guard abstains throughout that
held fold; it is not a learned detector of unseen zero-error events.

## Evidence and Decision

- 318,969 source targets: 311,922 ADE-supported, 7,047 unknown and 240,269
  endpoint-supported. Complete-future sensitivity retains 193,705 targets.
- Three seeds, three outer source folds, 12 already opened training localities;
  3,000 locality bootstrap resamples. No iid-window uncertainty claim.
- Complete metric reconstruction passes; all 90 heads reproduce 4,096 sampled
  rows each exactly. Nine groups have identical draws across six neural heads.
- Accounting verifies 12 reference views, 24 unchanged neural pointwise views
  and 144 decision receipts. Exact infeasibility pruning yields no solver
  failures and changes no old neural decisions across 72 receipt views.
- Independent selection, calibration and confirmation, including DroneCrowd,
  remain not_run. No deployment or submission promotion.

The result supports retaining a protected simple-motion control in every future
comparison. It does not support the proposed neural-trajectory or scene-joint
contribution yet. The next priority is to separate candidate error quality,
risk-moment reliability and nested producer-size shift before new fitting.
See [failure_analysis.md](failure_analysis.md), [execution_notes.md](execution_notes.md)
and [operation_zh.md](operation_zh.md).

This is released-detector, image-pixel, obs8/pred12 raw-stride12 source research,
not historical t50, seconds, meters, verified sensor-online ground truth,
physical safety, human gold, true 3D, foundation or submission-ready evidence.
Historical Stage37 remains exploratory after the later lineage audit.
Stage5C and SMC are off.
