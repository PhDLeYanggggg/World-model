# Matched Coverage Separates Ranking from Abstention

2026-09-21. `fresh_run`: fixed matched-policy arithmetic, independent selection
reconstruction and training-target feasibility audit. `cached_verified`: all
forecasts, cost heads, source geometry and labels. No new model was trained.
Independent calibration, untouched confirmation and deployment remain `not_run`.

## Answer

The asymmetric harm loss does **not** improve average ADE ranking over ordinary
MSE at the same intervention count. It does reduce observed harm to exact-zero-CV
queries, so its protection is not explained entirely by lower intervention rate.
This is a tradeoff, not domination or a safe deployment result.

The stricter budget also reveals an allocation problem: ranking by predicted
absolute net gain retains substantially more ADE improvement than ranking by
benefit/harm ratio at the same count. Every nontrivial control still fails the
strict zero-reference criterion. Neither a better average nor a positive-easy
diagnostic below 2% removes that failure.

## Fixed Experiment

Four already explored SDD source scenes, three seeds, 175,756 unique query
indices. The same frozen forecast is offered to every policy. Counts are fixed
to each scene/seed's existing underharm4 rule; all six policies choose exactly
K among candidate paths that differ from CV. Counts/ranks use no future outcome
or future-validity mask. This is an offline, full-source-batch diagnostic, not
an online prediction policy. Other queries in the batch can occur later in time;
the allocation is not claimed to be deployable causal streaming behavior.

The two mean intervention rates are 15.8109% and 0.8415%. No count, threshold,
seed or source scene is selected after this readout. The primary comparison is
underharm4 ratio versus MSE ratio; net-gain and query-shuffle controls are fixed
secondaries. All source data remain design-exposed, not independent tests.

## Low-Intervention Results

All rows below use exactly the same per-scene/seed count, averaging 0.8415%.
Task: 8 observed/12 predicted annotation steps, SDD pixel ADE against causal CV.

| Fixed Ranker | ADE Gain | Conditional Scene CI95 | Hard Gain | Positive-Easy Degradation | Zero-CV Harmed Instances | Max Zero-CV Harm (px) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Asymmetric ratio anchor | 0.3404% | [0.0188,0.8676] | 0.4015% | 0.1015% | 1 | 0.4503 |
| MSE ratio | 0.4347% | [0.0482,0.9670] | 0.5412% | 0.1347% | 15 | 9.9428 |
| MSE net gain | 1.1900% | [0.3236,2.3515] | 1.6552% | 0.4118% | 1 | 6.8364 |
| Asymmetric net gain | 1.1537% | [0.3191,2.2860] | 1.6022% | 0.3813% | 1 | 9.9428 |
| Signed ridge net gain | 0.6941% | [0.1872,1.2184] | 0.9485% | 0.3660% | 14 | 11.5457 |
| Fixed query shuffle | 0.0605% | [0.0047,0.1294] | 0.0799% | 0.2662% | 113 | 21.0999 |

Harmed instances repeat queries across seeds and are not independent people or
events. Hard q75 and positive-easy q25 remain training-defined diagnostics.
The zero-reference group uses complete future support; its percentage change
is undefined, so absolute errors are retained. All controls above fail that
strict check in at least one seed. MSE net gain selects 148 unknown-outcome
query/seed instances; the ratio anchor selects 48. Unknown is never zero risk.

The analytic expected gain for uniform selection of K eligible rows is 0.0632%,
close to the fixed shuffle's 0.0605%. Its expected zero-CV harmful-instance count
is 124.35. These are random-policy expectations, not realized trajectories,
guaranteed error bounds or tail quantiles.

## Higher-Intervention Results

All rows below average 15.8109% intervention, again exactly count-matched per view.

| Fixed Ranker | ADE Gain | Hard Gain | Positive-Easy Degradation | Zero-CV Harmed Instances |
| --- | ---: | ---: | ---: | ---: |
| Asymmetric ratio anchor | 4.4518% | 6.2350% | 3.8326% | 25 |
| MSE ratio | 4.5206% | 6.3897% | 4.0422% | 60 |
| MSE net gain | 4.7337% | 6.9012% | 4.8743% | 21 |
| Asymmetric net gain | 4.4518% | 6.2350% | 3.8326% | 25 |
| Signed ridge net gain | 4.4232% | 6.4011% | 5.1445% | 280 |
| Fixed query shuffle | 1.2287% | 1.7515% | 3.7186% | 1696 |

The uniform expected random gain is 1.1978%. At this budget, underharm4 has K
positive-net-score rows, so ranking its own positive net scores reproduces the
anchor exactly; this identity is expected, not a second independent success.
None preserves the positive-easy diagnostic within 2%, either.

## Paired Interpretation

The predeclared asymmetric-minus-MSE-ratio ADE contrasts are:

- Higher coverage: **-0.0688 percentage points**, conditional CI[-0.2313,+0.0352].
- Lower coverage: **-0.0943 percentage points**, conditional CI[-0.1592,-0.0294].

At lower coverage the mean contrast is negative in all four scenes; it is not
negative in every individual seed. The asymmetric objective trades accuracy for
fewer zero-reference harms. Its positive-easy advantage is small and uncertain:
lower-coverage contrast +0.0332pp, CI [-0.0235,+0.1208].

At that same lower count, MSE net gain exceeds the anchor by 0.8496pp,
CI [0.2654,1.4843]. This is a fixed secondary diagnostic, not a newly selected
deployment winner. All intervals average seed errors first and resample four
physical scenes 3,000 times. They are conditional developmental summaries;
multiple secondary contrasts and prior design exposure preclude a confirmatory
claim. Neither row overlap nor three seeds creates more independent scenes.

## Why the Ordering Matters

For a fixed K and additive predicted error, maximizing total predicted benefit
minus harm is achieved by selecting the K largest net gains. A ratio orders a
different quantity and can prefer a tiny opportunity with tiny harm over a much
larger expected improvement. For example, benefit/harm pairs (0.02,0) and (10,0.5)
give ratios 1 and 0.9048, but net gains 0.02 and 9.5. This mathematical observation
is not novel by itself and does not remove the second candidate's harm.

Accordingly, risk protection should be an explicit constraint or separate
decision objective, not inferred from an arbitrary benefit/harm ratio. The
current experiment does not validate such a replacement constraint.

## Training-Target Feasibility

A separate **training-only** audit of all twelve nested views finds 1,999-2,677
harmful complete zero-CV rows per view. Each view contains 7,012-10,001 complete
zero-CV rows in total. Counts are repeated and overlapping; they establish
target availability, not independent event counts or sufficient calibration.

Most zero-CV harms occur when the last observed step is zero: per view only 2-7
harmful zero-CV rows fall outside that causal subgroup. Nevertheless, that
subgroup also contains 3,943-6,058 beneficial examples, depending on producer.
An unconditional last-step-stop veto would therefore discard learnable gain,
and would not cover every zero-CV harm. Completely stationary eight-step histories
already receive unchanged forecasts: zero benefit, zero harm in these caches.
No new held outcome was used to set a veto or train a head.

## Next Action and Claim Limits

The evidence favors testing **expected net-gain ranking with separately learned
zero-reference/easy-harm risk**, using the clean nested views. Compare it with
the existing MSE and underharm4 heads at matched coverage. Train the new risk
target only from training supervision; do not supply future easy membership at
inference. Include the simple causal last-step-stop rule as a transparent
control, not an assumed solution. Freeze the comparison before new readout.

Independent calibration and confirmation remain separate missing evidence.
No closed original val/test, main/external or bookstore role was opened.
The old Stage26/37 results remain exploratory after lineage/exposure audits;
no new deployment is promoted here. Not metric, seconds-level, true3D or
foundation evidence. Stage5C execution and SMC remain disabled.

## Verification

All 392 parent/source/code bindings pass. 24 anchor results reproduce their parent.
All 144 policy/view counts match; a separate cutoff/partition implementation
reconstructs every selected-ID hash and independently checks 864 scene metric
reductions and 120 uniform-random expectation reductions. 37 scoped tests pass.
The full historical test suite was not rerun. Source arithmetic, not new model
training, is the fresh work. [Full table](results.csv), [analysis](analysis.json),
[independent verification](independent_verification.json),
[target support](risk_target_support.json), [reproduction](execution_notes.md).
