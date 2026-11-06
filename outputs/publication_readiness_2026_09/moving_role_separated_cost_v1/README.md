# Role-Separated Cost Learning And Full-Action Joint Decisions

I compared complete-bound anchor learning with separate observed-cost and
label-support supervision. All 48 Torch fits and all 48 complete exact refits
finished before policy readout. Both arms share architecture, fitting-only
normalization, initialization, candidate pool and complete source/query draws.
Reference mass and support probability are action-invariant by construction.

Dataset: MovingDroneCrowd++, already-exposed own-B. The original physical
floor is causal CV in these three contexts. The earlier observation-support
negative is retained; this is not a novel decomposition claim.

## Complete Development Policies

| Loss | Policy | Available-label gain % | Completion lower gain % | Supported views | Known easy >2% | Undefined easy |
|---|---|---:|---:|---:|---:|---:|
| composed_anchor | frame_uniform | +0.000061 | +0.000061 | 3/72 | 0 | 69 |
| composed_anchor | independent | +0.050017 | +0.047586 | 19/72 | 15 | 23 |
| composed_anchor | joint_free | +0.139664 | +0.134282 | 14/72 | 18 | 23 |
| composed_anchor | joint_matched | +0.122827 | +0.118608 | 16/72 | 20 | 23 |
| composed_anchor | proposal_only | +0.003738 | +0.003685 | 16/72 | 9 | 46 |
| role_separated | frame_uniform | +0.004273 | +0.003348 | 1/72 | 2 | 69 |
| role_separated | independent | +0.225466 | +0.181711 | 28/72 | 13 | 7 |
| role_separated | joint_free | +0.667496 | +0.587271 | 23/72 | 27 | 7 |
| role_separated | joint_matched | +0.616239 | +0.548201 | 22/72 | 25 | 7 |
| role_separated | proposal_only | +0.019827 | +0.019437 | 11/72 | 17 | 43 |

72 views repeat 24 recordings over three risk-head seeds; they are not
72 independent scenes. Eight source blocks support exploratory uncertainty.
Forecast teacher seed remains17. Gains use available-label native obs8/pred12
image-coordinate ADE, not official complete-12 ADE or verified seconds/meters.

## Registered Primary: Role-Separated Minus Anchor, Joint-Free

| Contrast | Point pp | Exploratory paired source interval |
|---|---:|---:|
| completion_lower_pp | 0.445303 | [+0.090835, +0.930796] |
| completion_upper_pp | 0.610360 | [+0.277596, +1.062217] |
| known_pp | 0.527832 | [+0.193976, +0.996965] |

3,000 source-component resamples after averaging risk seeds within source.
The paired unknown envelope is between policies; separate lower gains cannot
be subtracted to claim a paired improvement. Secondary controls do not replace
the registered primary. No independent C/H evaluation is opened.

All three risk-seed completion-lower contrasts are positive; six of eight
source means are positive. Source065 has a negative lower contrast, and
source073 is negative even on known labels. These are not three independent
forecast-teacher replications. Full values are in primary_source_seed_rows.json.

## Mechanism: Matched Prediction Targets

| Loss | Observed conditional MSE | Observed composed MSE | Bound composed MSE | Label-support AUROC | ECE |
|---|---:|---:|---:|---:|---:|
| composed_anchor | 1.348588 | 1.415037 | 7.063742 | 0.539627 | 0.058111 |
| role_separated | 1.189763 | 1.308119 | 7.661082 | 0.679486 | 0.038312 |

Both arms use the same known-only five-moment fitting RMS. These numbers
are not comparable to earlier two-channel signed-MSE tables. Bound error is
against conservative contributions, not ground-truth unobserved outcomes.
Support scores predict released-label availability, not human visibility or
baseline failure. Full source/seed data, AUPRC/Brier and the separately matched
old-proposal signed-error control are retained in components.json.

The new composed outputs and the old known-only outputs have different
statistical targets. A separate conditional-risk probe therefore compares
the conditional branch on the same known old-proposal cohort and RMS:

| Model | Conditional signed MSE |
|---|---:|
| Frozen known-only head | 1.609501 |
| composed_anchor conditional branch | 13.721483 |
| role_separated conditional branch | 2.810864 |

Role separation improves this new model family, but does not recover the
old head's conditional error on this matched cohort. This posthoc probe
does not select a new checkpoint or replace the registered utility contrast.

## Selected-Risk Calibration

| Policy | Pass | Known harm violation | Completion-only violation | Missing support |
|---|---:|---:|---:|---:|
| composed_anchor/joint_free | 14 | 18 | 17 | 23 |
| role_separated/independent | 28 | 13 | 25 | 6 |
| role_separated/joint_free | 23 | 27 | 16 | 6 |

Categories are mutually exclusive; observed violations take priority over
completion-only and support failures. Seven undefined easy views in the
main table need not equal six support-only failures in this table.
The separated joint rule overpredicts selected easy reference in 61/72 views
and underpredicts selected easy harm in 35/72. Seven easy violations select
no unknown rows. This implicates selected-cost/reference optimism, not only
censoring. Independent-to-joint improves utility but reduces full support
from 28/72 to 23/72. Predicted query feasibility is not realized safety.

## Candidate Coverage And Same-Count Controls

- composed_anchor / independent_minus_proposal_only: paired lower 0.043901 pp; interval [0.013217588005480107, 0.08096398503059907].
- composed_anchor / joint_free_minus_independent: paired lower 0.085476 pp; interval [0.018708133312749682, 0.17297478507925218].
- composed_anchor / joint_matched_minus_independent: paired lower 0.069647 pp; interval [0.011654035432462934, 0.14801837076016394].
- role_separated / independent_minus_proposal_only: paired lower 0.162274 pp; interval [0.013709589613088904, 0.35285761113647884].
- role_separated / joint_free_minus_independent: paired lower 0.402450 pp; interval [0.1348857734375232, 0.7985025012995121].
- role_separated / joint_matched_minus_independent: paired lower 0.361559 pp; interval [0.10412853779359614, 0.7438922475125708].

Matched-count utility is computed only on causally selected nonempty
queries and is never used for safety. Empty controls remain not estimable.
Complete-policy safety evaluation retains all required recording views.

## Verdict

Registered development candidate screen: **FAIL / NA**.
No automatic deployment or submission-ready claim. The full original2%
positive-harm, signed easy-net and reference/support requirements remain.
Empty predictions receive no identity exception. Earlier negatives remain.

fresh_run:full-action data/moments,48fits,48fullrefits,causal allocation/readout.
cached_verified:forecasts,old proposals,source boundaries and legacy controls.
not_run:new independent C/H,three forecaster seeds,deployment or new dynamics.
The original24-group main stays NA. No metric/seconds,true3D,foundation,
human-gold,Stage5C or SMC claim. Annotation-generation causality unverified.

[Failure analysis](failure_analysis.md) | [Model card](model_card.md) |
[Reproduction](operation_zh.md) | [Verification](verification_report.md) |
[Paper addendum](paper_addendum.md)
