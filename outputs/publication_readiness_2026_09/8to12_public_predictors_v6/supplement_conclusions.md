# Frozen Forecast Supplements: Complete, No Stable Joint Gain

Date: 2026-09-17. Both model families, seeds 17/29/43, both cost heads and both
prespecified policies were evaluated. No supplementary result selects a model
or changes a threshold. The failed primary selections remain CV.

## Exact Raw-Frame t+50

The two development recordings have a ten-frame annotation grid. Raw50 therefore
scores the first five steps of the original twelve-step forecast, with its
original past normalization. It is not a separately conditioned t+50 model,
an interpolation, or a seconds-level result. The prefix has 33,686 complete ADE
paths and 33,690 valid endpoints, versus 28,324 complete paths for the primary
task. Unknown full-path easy/hard labels remain unknown, not assigned as easy.

| Model | Seed | Uncontrolled raw50 ADE gain vs CV | Uncontrolled raw50 FDE gain vs CV | Neural-cost conservative joint raw50 ADE gain |
| --- | ---: | ---: | ---: | ---: |
| Transformer | 17 | -21.643% | -10.077% | -0.02529% |
| Transformer | 29 | -22.289% | -12.004% | -0.01968% |
| Transformer | 43 | -15.238% | -9.496% | -0.00588% |
| EqMotion-K1 | 17 | -50.961% | -9.214% | -0.03895% |
| EqMotion-K1 | 29 | -35.881% | -13.585% | -0.18764% |
| EqMotion-K1 | 43 | -61.229% | -46.882% | -0.15759% |

The displayed conservative policy is prespecified, not chosen as the best
supplement. Every joint policy has negative raw50 ADE gain; the full tables
retain all arms, including occasional positive FDE values that do not reverse
the negative ADE/easy result. Floor and scene-uniform controls return zero gain.

On the full-path-defined easy slice, prefix CV ADE is 0.001532864. The displayed
neural-cost joint policy has easy ADE 0.003586/0.001966/0.004008 for Transformer
and 0.005753/0.019210/0.016108 for EqMotion (seed order 17/29/43). Both absolute
and relative damage remain reported; the small baseline denominator is not an
excuse to suppress easy degradation.

## Same Actual Intervention Count

Independent and joint controls use identical candidates and the same realized
switch count per past-supported scene query. Counts are fixed before labels
are accessed. This does not imply equal coverage after missing labels are
filtered, equal realized harm, or formal safety.

Transformer has zero joint-minus-independent ADE and FDE difference for all
twelve head/policy/seed combinations, including matched nonzero queries. The
switch identities also match on every exported agent query. Thus its zero
difference is not merely cancellation of positive and negative errors.

EqMotion has zero difference for ten combinations. Seed29 ridge conservative
has joint-minus-independent ADE +0.00000904878 (worse); ridge moderate has
-0.0000124400 (better). FDE signs agree. Across the twelve reused candidate
exports, only 76 of 453,300 agent-query decision records change switch identity.
These repeated exports are not 453,300 independent observations. Nonzero
matched scene-query counts range 121--968 per combination; no solver/unmatched
query is omitted. This tiny, directionally inconsistent effect does not support
a robust joint-intervention contribution.

## Replay and Statistical Scope

For each family, all 24 candidate/recording replay parts pass source hashes and
completion checks. Every exported original floor/candidate error and ordinary
control decision matches the primary evaluation exactly. This verifies recorded
errors/decisions, not bitwise identity of every intermediate model tensor.

The two recordings still represent one physical development site. Scene CI is
explicitly not_run for insufficient independent scenes; training-seed SD does
not replace that uncertainty. Source clocks, geometry and annotation lineage
limits are unchanged. There is no new deployment, Stage5C, SMC, physical-safety
or submission-readiness claim.

## Full Evidence

- [Transformer supplement](../8to12_transformer_v6_supplement/results.md)
- [EqMotion supplement](../8to12_eqmotion_v6_supplement/results.md)
- [Transformer replay audit](../8to12_transformer_v6_supplement/replay_audit.json)
- [EqMotion replay audit](../8to12_eqmotion_v6_supplement/replay_audit.json)
- [Primary conclusions and next hypothesis](conclusions.md)
