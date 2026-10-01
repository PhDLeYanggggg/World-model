# Nonlinear Cost Regression: Better Preservation, No Safe Utility Upgrade

Result source: **fresh_run**, with exact first-fit and complete readout replay.
Inputs and neural comparators: **cached_verified**. Independent confirmation:
**not_run**, closed by the existing protocol. Deployment is unchanged.

This is a conventional ExtraTrees control, not new neural dynamics training.
The data are12 already exposed European development localities, image-local
detector-silver, obs8/pred12 at stride12 raw frames. No metric, seconds, true3D,
foundation, human-gold or physical-safety claim. Stage5C and SMC remain off.

## What Ran

72 source models, three seeds17/29/43,128 trees/model,9,216 trees total.
Every source optimization/validation split is recording-disjoint. Optimization
preprocessing was recomputed and exactly matched to the frozen neural comparator.
The target transformation matches its composite quadratic moment/signed-score
objective before projection. Features contain the same causal information, but
the tree receives envelope as an explicit input and projects outputs afterwards.
Finite sampling and decoder placement differ: this does not isolate architecture
or optimizer effects perfectly. See [protocol](protocol.md).

44/72 forests have lower source-validation signed-score MSE than the selected
neural checkpoint. However, the locality-balanced primary difference is not
favorable overall:

| Source-validation comparison | Value |
|---|---:|
| Forest signed-score MSE |1.133789|
| Validation-selected neural MSE |1.112728|
| Final neural MSE |1.483433|
| Initial-prior MSE |1.173913|
| Forest minus selected neural |+0.021061|
| Nominal locality95% interval |[-0.020329,+0.069864]|

The interval crosses zero. There is no supported source-validation superiority.
28/72 forests pass the fixed source finite-completion screen:8/24 for seed17,
9/24 for29,11/24 for43. No transfer labels selected these models or thresholds.

## Frozen Cross-Scene Readout

All216 causal prediction/action views were committed before outcome readout.
The secondary same-query-count ADE improvement of forest over neural is
**-0.148667%**, nominal locality95% interval **[-0.252426%,-0.060646%]**.
The three seed means are -0.165444%, -0.148065%, -0.132494%;11/12 locality means
are negative. The tree selects less useful interventions at the same count.

Transferred signed-score MSE is lower on average by0.200706, but its nominal
interval[-0.450370,+0.003421] also includes zero. Global cost approximation does
not establish decision utility or selected-tail risk calibration.

All gains below are percentages against the frozen floor. Intervention is a
percentage of rows. Risk failures use the unchanged2% selected-reference budget,
not the full-population easy ADE denominator.

| Policy | ADE gain | Hard gain | Intervention | Worst whole-easy degradation | Easy risk failures/defined | Undefined |
|---|---:|---:|---:|---:|---:|---:|
| Forest |0.052957|0.000407|4.559529|0.032322|17/140|76|
| Neural selected checkpoint |0.568087|0.615400|16.772359|21.054762|149/195|21|
| Initial prior |0.283623|0.003914|16.844171|9.135868|126/177|39|
| Forest matched count |0.048149|0.000337|4.122131|0.031510|16/128|88|
| Neural matched count |0.193514|0.164045|4.122131|1.178199|58/125|91|
| Source-screened forest |0.048206|0.000404|4.047939|0.032322|6/68|148|
| Floor only |0|0|0|0|0/0|216|

The screened forest's ADE improvement is0.048206%, interval[0.029848%,0.068106%].
Its FDE improvement is0.072122%. These tiny exposed-development gains do not
override selected-risk failures: all-risk fails4/68, easy risk6/68, with worst
easy selected positive-harm ratio7.550605%. Selected unknown outcomes occur1,953
times across overlapping/repeated evaluation views, not1,953 independent cases.
148 undefined views are not passes. Floor-only undefined risk is likewise not a
positive safety certificate.

All six screened easy-risk violations occur in seed43, across several source/
target pairs. Seeds17 and29 have0/17 and0/21 violations among their defined views,
while43 has6/30. This is descriptive, not permission to discard seed43. The
shared seed also affects upstream predictors/controllers, so forest randomness
has not been isolated. See [failure analysis](failure_analysis.md).

## Verification and Compute

- 30 scoped tests passed; full legacy suite **not_run**.
- First128-tree fit exactly retrained: trees, predictions, input hashes and
  traces agree, excluding elapsed time. All72 were not retrained twice.
- Full216-view readout exactly replayed with all frozen action hashes preserved.
-37,800 independently reconstructed metric values;936 independent cost-score
  checks;747,900 per-query intervention-count checks.
- Three training seeds and3,000 locality bootstrap resamples. Intervals are
  nominal development diagnostics, not independent population evidence.
-496.78 cumulative fitting seconds;161,406,781 checkpoint bytes.
- First evaluation240.11s; exact replay239.19s; first-fit replay51.58s.
- Pilot peak RSS4,838,981,632 bytes. CPU4, interop1, workers0, native arm64.
- Original10GiB reserve kept. A structure-based preflight amendment resolved an
  overly conservative storage estimate without changing scientific settings.
- Registration418b27e5; resource amendment1017ae5c; models c8dfb7ce;
  actions a4b20803, all committed before their dependent phases.

Numeric/source verification seal:
`aac8bc86b0139d06866a323ee08112d57376c7cb50ab6dd5597fee3ca3de35a9`.
It binds108 code/config/protocol files and16 artifacts. Later narrative delivery
has a separate seal; the original numeric seal is not rewritten.

## Verdict

**No deployment promotion and not yet a submission candidate.** The estimator
control ran successfully and improves preservation relative to the unfiltered
neural comparator, but loses matched-count utility and still violates selected
risk. It does not show that causal features are useless or that all neural
optimization is sound. Nor does a conventional forest become a neural world-model
contribution. Keep the frozen floor and the original scientific restrictions.
