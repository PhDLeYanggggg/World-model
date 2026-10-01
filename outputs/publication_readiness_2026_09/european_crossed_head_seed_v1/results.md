# Fixed Upstream, Different Cost-Head Seeds: Risk Failures Persist

**No deployment promotion.** Changing only the cost-head random seed does not
repair selected-tail risk. There is no supported same-count utility advantage
over the original43 head. This is a completed estimator diagnostic, not new
neural dynamics training or an independent generalization success.

## Evidence and Scope

- **fresh_run:**48 new128-tree ExtraTrees fits,72 frozen directional readouts
  for all three heads, component accounting, bootstrap and exact replay.
- **cached_verified:**24 original43 cost heads, upstream predictors/controllers,
  features, partitions and original comparisons, with hashes checked.
- **not_run:** independent selection/calibration/confirmation and deployment;
  all48 fits were not retrained twice; the full legacy test suite was not run.

The experiment keeps upstream43 fixed and varies head seeds17/29/43. Input,
target, weighting, mask, preprocessing, source-recording splits and tree settings
are identical. The 48 new fits comprise6,144 trees; all72 retained heads comprise
9,216 trees. They are not three independent upstream training replicates.
See the [registered protocol](protocol.md).

The data are12 already exposed European development localities, obs8/pred12 at
stride12 raw frames, image-local detector-silver. No metric, seconds, human-gold,
true3D, foundation or physical-safety claim. Stage5C execution and SMC stay off.

## Utility and Selected Risk

All gains and intervention rates below are percentages. Results are equal-locality
development means. The source screen is fixed before transfer and uses only
source validation. The2% positive-harm budget has the selected-reference
denominator, not the whole-population easy ADE denominator.

| Head | Policy | ADE gain vs floor | Hard gain | Intervention | Worst whole-easy degradation | Easy risk failures/defined | Undefined |
|---|---|---:|---:|---:|---:|---:|---:|
|17|raw|0.052712|0.000457|4.771788|0.008740|10/46|26|
|29|raw|0.051847|0.000505|4.788901|0.011816|8/47|25|
|43|raw|0.053216|0.000285|4.774383|0.032322|9/47|25|
|17|source-screened|0.047661|0.000437|4.222812|0.003339|7/30|42|
|29|source-screened|0.046821|0.000480|4.240320|0.008631|5/30|42|
|43|source-screened|0.048333|0.000277|4.275661|0.032322|6/30|42|

Source-screened ADE gain nominal95% intervals are[0.031085%,0.064938%],
[0.029977%,0.064294%] and[0.031539%,0.065531%], respectively. Screened FDE gains
are0.070963%,0.069749%,0.071535%. The hard gains are tiny; head43 hard-gain
interval[-0.000076%,0.000667%] crosses zero. These are not substantial neural
dynamics gains. The parent43 averages here use only upstream43 contexts, not
all upstream seeds of the preceding experiment.

All-risk violations remain5/30,3/30,4/30; worst easy selected risk is4.5322%,
4.5322%,7.5506%. Unknown selected occurrences are734,743,757 across repeated
views, not independent samples.42 undefined directions per head are not safety
passes. None of the three heads is a safe upgrade.

## Same-Count Comparison

The paired controls match intervention counts separately within each query.
They avoid confusing a coverage change with improved selection.

| Head vs43 | Matched ADE improvement | Nominal95% locality CI | Transferred signed-MSE difference |
|---|---:|---|---:|
|17|+0.000041%|[-0.000881%,+0.000922%]|+0.001221|
|29|-0.000162%|[-0.000730%,+0.000414%]|+0.002430|

Neither utility interval excludes zero. Cost MSE is not improved: head17 MSE
difference interval[-0.000631,+0.003212] crosses zero, while head29 interval
[+0.000667,+0.004352] indicates worse fit in this nominal development contrast.
Source-validation signed MSE is1.119960,1.120566,1.119043, and every seed passes
11/24 source screens. No seed winner is selected.

Intervals use3,000 resamples of12 locality means, not individual overlapping
windows. Localities and the motivating failures were already exposed in prior
development. These intervals are conditional/descriptive, not independent
population guarantees or confirmatory evidence.

## Why the Risk Gate Still Fails

Three of the six parent failures persist under all three raw heads. Two persist
after every source screen; a third disappears for one seed only because its
source is rejected. Changing a seed also creates new failing directions.

The component diagnostic identifies both underestimated harm and overestimated
easy-reference error. In007 ->124, all heads select the same4.5322%-risk outcome;
predicted reference is around0.68-0.70 but observed reference is0.1956. Predicted
harm is already conservative, so harm-only penalties miss the denominator
problem. Conversely,074 ->112 is dominated by underestimated harm.
See the [full six-case analysis](failure_analysis.md).

This weakens a cost-head-seed-only explanation. It does not uniquely identify
an upstream component, prove that all features are inadequate, or authorize
discarding seed43. Source component calibration/support is the next hypothesis
to test, not a claimed solution.

## Verification and Runtime

- First full128-tree new fit replayed exactly, including trees and predictions.
- Entire792-row,216-head-view readout replayed exactly.
- All216 original43 raw/screened/floor metric views agree exactly with the parent.
-19,800 independently reconstructed native metric values and498,600 query-count checks.
-34 scoped tests passed; full legacy suite not_run.
-372.824561 cumulative fitting seconds;106,973,492 new checkpoint bytes.
- Evaluation127.09s; exact replay124.56s; first-fit replay79.63s.
- Native arm64 CPU4/interOp1/workers0; original10GiB reserve preserved.
- Registration `b0c32be1`; model freeze `abed7896`; action freeze `dfa00956`.

Numeric seal `6731eb97347a8defcd8abda080b6179d1923e21c9b8d0e80d79bf0a5640e4e71`
binds110 source/config/protocol files and13 artifacts. Narrative/state delivery
is separately checked. [Reproduction and recovery](operation.md).

**Verdict:** a valid completed negative control, not a deployable model or a
submission-ready main result. Retain the frozen floor and closed independent
roles. Do not replace this conclusion with the best-looking seed.
