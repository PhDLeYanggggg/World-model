# Harm Allocation: What Changed And Why It Failed

## Result

The fixed nine-fit experiment is complete, not successful for deployment.
Allocating more training weight to lower-magnitude harmful proposals reduces
false-safe fitting decisions, but the lost useful interventions cost more than
the recovered benefit. The primary exposed-development utility contrast is
negative and the original safety/support requirement fails.

This is a native obs8/pred12 image-coordinate intervention-risk experiment.
It does not train a new trajectory forecaster or establish new world dynamics.
Parent data and models are cached_verified; fits, readout and attribution are
fresh_run. Independent reserved/H confirmation is not_run.

## Controlled Change

Three own-B contexts each train joint, uniform_harm and harm_floor for 2,000
updates. Inputs, preprocessing, warm starts, sampler draws, unknown-label
supervision, original risk MSE, last-iterate rule and inference thresholds are
fixed. The latter two arms have the same total positive coefficient mass under
the exact planned query weights. Equal mass does not mean equal realized
gradients or equal loss. The harm floor is the own-B weighted median positive
target magnitude; it is not an inference threshold.

All three joint controls reproduce the preceding models exactly. All nine
sampler/weight calculations and initial/final metrics replay exactly. One
complete treatment refit and three interrupted-versus-continuous pilots are
exact. A receipt serialization failure was repaired without changing model
state or scientific bindings; the failed attempt is retained.

## Fitting Mechanism

| Context | Uniform false-safe | Floor false-safe | Uniform false-reject | Floor false-reject | Floor joint-cost change |
|---|---:|---:|---:|---:|---:|
| critic0 | 17 | 7 | 2660 | 2693 | +0.6261% |
| critic1 | 32 | 11 | 1942 | 2001 | +0.2402% |
| critic2 | 143 | 81 | 22752 | 22971 | +0.3512% |

The intended suppression works, but discrimination does not improve enough.
Against the equal-mass uniform control, false-safe cost falls 43.53%, 77.24%
and 40.28%; false-reject cost rises 1.05%, 2.49% and 1.25%. Joint cost increases
in every context. The original internal screen still passes 3/3 because it
compares with warm initialization, not the matched fitted control or held safety.

After freezing the readout, I decomposed the decisions on own-B only. This
posthoc descriptive diagnostic is separately registered and exactly replayed;
it trains nothing and reads no new held outcomes.

| Uniform to floor | critic0 | critic1 | critic2 |
|---|---:|---:|---:|
| Harmful admissions repaired | 10 | 21 | 62 |
| Harmful admissions retained | 7 | 11 | 81 |
| Newly harmful admissions | 0 | 0 | 0 |
| Useful admissions lost | 75 | 64 | 227 |
| Useful admissions recovered | 42 | 5 | 8 |
| Lost benefit, weighted cost | 0.00017884694 | 0.00009223776 | 0.00026875493 |
| Repaired harm, weighted cost | 0.00002707551 | 0.00007030693 | 0.00017158458 |
| Recovered benefit, weighted cost | 0.00011126128 | 0.00001417566 | 0.00002834257 |
| Net joint-cost increase | 0.00004051015 | 0.00000775516 | 0.00006882777 |

The last line equals lost benefit minus repaired harm minus recovered benefit;
the exact values match the existing fit receipts. Counts are not substitutes
for weighted cost. The slices explain a training tradeoff, not generalization
causality. Weight construction averages positive channels; attribution sums
their magnitudes. Absolute values from those two conventions are not identical.

## Exposed Development

| Policy | Known ADE gain % | Hard gain % | Switch % | Worst defined easy-net % | Original full risk/support |
|---|---:|---:|---:|---:|---:|
| Joint | 0.894772 | 1.169812 | 5.988241 | 8.633364 | 0/17 |
| Uniform | 0.880858 | 1.158954 | 5.904828 | 8.633364 | 0/17 |
| Floor | 0.792581 | 1.046007 | 5.395710 | 8.048131 | 1/17 |

Floor minus uniform is -0.088277 pp, with missing-label completion bounds
[-0.100385, -0.076169] pp. These are **not sampling confidence intervals**.
Six of seven SOURCE units lose known utility. SOURCE075's small positive
contrast does not justify posthoc selection and its absolute utility remains
negative. These are seven previously exposed development sources, 17 recording
views and one producer seed, not independent confirmation. Known partial-outcome
ADE is not complete-roster 12-step ADE.

The floor still has 12 known all-risk and 13 known easy-risk violations.
SOURCE062 retains the worst defined signed easy-net degradation, 8.0481%.
Only SOURCE077 recording47 satisfies the original full risk/support check.
Three selected-easy views remain undefined: SOURCE066/33, SOURCE070/38 and
SOURCE075/43. Neither zero net change nor an empty reference creates a pass.
The original 24-group primary remains NA. Positive-harm risk and signed easy-net
are different quantities; both remain governed by the original requirements.

Removing the decision veto raises known gain to 3.303746% but worst easy-net
harm to 40.377560%. That diagnostic is not a repair. No deployment is changed.

## Failure Taxonomy

- Training execution: repaired and verified; not the explanation for the negative contrast.
- Inadequate harm penalty allocation: suppresses more harmful decisions, but does not solve useful/harmful separation.
- Fitting-versus-development mismatch: the internal screen is insufficient for scientific safety, not a reason to rewrite it retrospectively.
- Reference/support limitation: unresolved and distinct from actual observed harm violations.
- Feature or conditional representation insufficiency: plausible, not uniquely proved by this experiment.
- Simple benefit-mass dominance: not supported by the preceding own-B attribution; do not revive it as the explanation.

## Next Model Work, Not Yet Run

1. On own-B only, compare causal representations of lost useful, repaired harmful
   and retained harmful proposals, including action family and source slices.
   Separate representation overlap from optimizer or target-estimation errors.
2. Register one causal representation or paired decision-learning change with
   the current equal-mass control. Target useful/harmful separation, rather than
   globally increasing rejection. Keep future truth exclusively in supervision.
3. Retain the original evaluation contract and a frozen readout. No threshold
   search on these outcomes and no reserved data access to rescue the result.

No identity-fallback certification, metric/seconds/true3D/foundation/human-gold
claim, Stage5C execution or SMC. This remains not yet a submission candidate.
