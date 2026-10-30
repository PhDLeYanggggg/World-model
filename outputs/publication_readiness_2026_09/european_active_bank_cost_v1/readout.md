# Gain/Harm Learning On Matched Frozen Active Banks

## Material Passport

Fresh_run: 54 real Torch cost heads, 108,000 primary updates, 2,000 full-
refit updates, 160 pilot updates and a complete separate policy replay.
Cached_verified: original SOURCE inputs, original cost controls and frozen
seed43 trajectory producers. Three new past-only proposal caches were
built in this round; they are frozen inputs, not new trajectory training.
Not_run: controller-held outcomes, independent calibration/confirmation,
new scene-joint decision training or deployment.

Training PID71099: 282.964357s, peak RSS 11,008,163,840B.
Readout PID73164: 126.493666s; separate exact replay PID74514: 125.213123s.
All 36 old cost endpoints reproduce their full scientific states. Old
moments, actions and required-recording risk scores replay exactly in all
72 groups. The first new-bank head has an exact 2,000-step independent
refit. Native arm64 CPU4/workers0; no NumPy training fallback or budget cut.

## Same Learner, One Replaced Candidate

All six physical candidates, the causal floor, original past-only coverage
cohort, draws and physical/detector fitting preprocessing stay fixed.
Only the seventh candidate changes. Its motion/region features, envelope
and supervised costs are rebuilt together through separate causal/label
interfaces. Cross-bank comparisons use actual paths, not stale action6 costs.

Unique fitting queries: 149,510; recordings: 74; localities: 12.
Known: 146,347; wholly unknown: 3,163; repeated row views: 897,060.
Repeated windows/context/seed views are not independent samples. Intervals
use 3,000 locality resamples after context/head-seed averaging. SOURCE has
historical development exposure: these are fitting outcomes, not a new test.

## Complete-Policy Accuracy And Risk

| Bank | Known ADE gain vs floor %, CI95 | FDE gain %, CI95 | Recording risk supported | Known easy-risk violations | Undefined easy-risk views | Switch row views |
|---|---|---|---:|---:|---:|---:|
| physical_only | +1.946059 [+0.714940, +3.800027] | not_estimable | 42/444 | 313 | 35 | 268083 |
| old_neural | +1.968657 [+0.727097, +3.864063] | not_estimable | 42/444 | 307 | 33 | 259875 |
| new_neural | +1.893446 [+0.692104, +3.721837] | not_estimable | 37/444 | 313 | 32 | 267092 |

Risk support uses the unchanged 2% complete-policy all/easy rule on
every required recording, including missing-outcome bounds. Raw easy mean
gain and positive selected-easy risk are different quantities. Empty risks
are not_estimable, not safe zero. Complete-roster FDE stays undefined when
a required locality has no requested final labels.

## Paired Candidate-Bank Contribution

| Contrast | Full known pp, CI95 | Full completion-lower pp, CI95 | Full completion-upper pp, CI95 | Same-count lower pp, CI95 |
|---|---|---|---|---|
| new_minus_old | -0.075211 [-0.171084, +0.012147] | -0.681429 [-1.441316, -0.218155] | +0.531006 [+0.032367, +1.318329] | not_estimable |
| new_minus_physical | -0.052613 [-0.161055, +0.046791] | -0.571283 [-1.126993, -0.226855] | +0.466057 [+0.050698, +1.091184] | not_estimable |
| old_minus_physical | +0.022598 [-0.032990, +0.090980] | -0.301435 [-0.520784, -0.131315] | +0.346632 [+0.132836, +0.631789] | not_estimable |

Differences retain the same known floor-reference mass, not a newly
selected denominator. Negative completion bounds are not observed damage.
Same-count controls assess utility only. Empty comparisons retain undefined
points/intervals and are not credited as passing safety controls.

## Does The New Head Actually Use Neural Trajectories?

The same frozen new head is also decoded with its neural action disabled,
without refitting. This secondary control separates neural trajectory use
from changes in physical-action decisions due to shared-head training.

- Neural-selected repeated row views: 88379
- Unknown neural-selected views: 940
- Full known neural-use increment pp: +0.165906 [+0.054680, +0.334448]
- Full completion-lower neural-use increment pp: +0.125674 [-0.005768, +0.302693]

## Future-Informed Opportunity, Not A Learned Policy

| Bank | Required context/record views, seeds deduplicated | Feasible | Positive feasible |
|---|---:|---:|---:|
| physical_only | 148 | 148 | 148 |
| old_neural | 148 | 148 | 148 |
| new_neural | 148 | 148 | 148 |

The shared union-oracle regret is diagnostic and has no inference role.
The union is not an eight-action deployment bank. Bank-local regrets are
preserved for legacy equality, not compared as one common oracle.

## Real Training Objective

| Bank | Heads | First median minibatch objective | Final median minibatch objective |
|---|---:|---:|---:|
| physical_only | 18 | 2.08923 | 0.333939 |
| old_neural | 18 | 1.8956 | 0.324785 |
| new_neural | 18 | 1.90402 | 0.332831 |

Loss is asymmetric five-moment cost regression plus within-row ranking,
not best-class classification. Curves are actual fitting minibatches, not
validation or independent-test losses. Unknown labels receive no observed-
cost loss and do not determine the common row sampler.

The [selected-moment diagnostics](moment_diagnostics.json) also report
reference-error bias and harm underprediction on known selected rows.
They are repeated-view fitting diagnostics, not probability calibration
or a proven causal explanation of failure.

## Fitting-Progress Gates

- full54fits_and72groups: true
- legacy36endpoints_and72readouts_exact: true
- complete_new_policy_recording_risk: false
- new_minus_old_known_CI_positive: false
- new_minus_old_completion_lower_CI_positive: false
- matched_new_minus_old_completion_lower_CI_positive: false
- new_minus_physical_completion_lower_CI_positive: false
- fitting_advancement_gate: false
- independent_confirmation: false
- deployment_changed: false
- submission_ready: false

SOURCE is supplied detector-silver bbox-midpoint pixels, obs8/pred12 at
stride12 raw frames. Not historical SDD t50, human gold, verified live
sensor-as-of, metric/seconds, true3D or foundation evidence. No Stage5C
execution or SMC. A complete anonymous raw-data/weight release and the
scene-joint/independent-evidence requirements are not fulfilled here.

[Actual curves](figures/summary.svg), [machine evidence](evidence.json),
[failure analysis](failure_analysis.md), [model/data card](model_data_card.md),
[Chinese guide](reproduction_zh.md).
