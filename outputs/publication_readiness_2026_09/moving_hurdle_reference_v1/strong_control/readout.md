# Comparison With The Existing Learned Neural Reference

Fresh replay, no new fitting or decisions. All108new reference arrays and18old neural
cost arrays match their completed evaluation hashes; both estimands and all23supported
source groups replay exactly. Three actual seeds; 3000source-group bootstrap resamples
after producer/seed averaging. These are exposed-development diagnostic intervals,
not independent confirmation or multiplicity-adjusted claims.

|Estimand|New reference minus old neural: MSE reduction [95%CI]|
|---|---:|
|conditional_known: C_selected|-0.000142915 [-0.000222749,-7.55963e-05]|
|conditional_known: direct|-0.000140965 [-0.000221001,-7.3533e-05]|
|conditional_known: hurdle|-0.000163075 [-0.000255607,-9.09073e-05]|
|conditional_known: prior|-0.000211723 [-0.000354065,-0.000108892]|
|original_bounded_label: C_selected|-0.000143482 [-0.000221417,-7.79073e-05]|
|original_bounded_label: direct|-0.000141512 [-0.000220349,-7.54496e-05]|
|original_bounded_label: hurdle|-0.000163252 [-0.000254426,-9.25068e-05]|
|original_bounded_label: prior|-0.000213207 [-0.000353116,-0.000112653]|

Positive means lower normalized ER prediction MSE, not an ADE/FDE percentage.

C-selected improvement over the learned control is established on known labels: False.
C-selected improvement over the learned control under original bounded labels: False.
Old neural reference is better on both targets with intervals excluding zero: True.

**Conditional-known** scores observed eligible rows. **Original-bounded-label** keeps all
eligible rows and the old conservative zero reference contribution for wholly unknown
labels. That zero is not evidence of a negative physical future event. No future label
availability enters inference. The inherited neural R/ER already shares its reference
across all seven candidates; reference sharing is not a new repair.

The new heads train on known-label ER only. The old neural head fits multiple bounded
moments, including unknown-reference-zero contributions. These are actual operational
comparators, not matched input, architecture, target or capacity ablations. A loss contrast
cannot identify which difference causes it or justify substituting a conditional mean
into the bounded deployment decoder.

C-only selection remains16direct/2hurdle. No held-source model or threshold reselection.
All prior support failures, original2%positive-harm/support and signed easy-net conditions
remain. The original24group primary is not_estimable. Reserved Moving20groups, DUT and
DroneCrowd confirmation stay closed. No new policy, trajectory gain, record-risk guarantee
or submission readiness. No metric/seconds/true3D/foundation/human-gold claim; Stage5C/SMC off.

## Absolute Prediction Losses

|Estimand|Model|Source-group mean normalized MSE [95%CI]|
|---|---|---:|
|conditional_known|prior|+0.000775022 [+0.000498095,+0.00110393]|
|conditional_known|direct|+0.000704265 [+0.000457323,+0.000981221]|
|conditional_known|hurdle|+0.000726374 [+0.000472228,+0.00101317]|
|conditional_known|C_selected|+0.000706215 [+0.000458312,+0.000984431]|
|conditional_known|old_neural|+0.0005633 [+0.000343182,+0.000801948]|
|original_bounded_label|prior|+0.000765773 [+0.000493815,+0.00108924]|
|original_bounded_label|direct|+0.000694077 [+0.000452257,+0.000966623]|
|original_bounded_label|hurdle|+0.000715817 [+0.000464681,+0.000999922]|
|original_bounded_label|C_selected|+0.000696047 [+0.000452296,+0.000969153]|
|original_bounded_label|old_neural|+0.000552565 [+0.000336622,+0.000788088]|
