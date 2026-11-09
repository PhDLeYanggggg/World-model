# Matched direct forecasting on OpenDD

I compared the velocity spline with CV, OLS4, and a frozen fixed-head
[EqMotion](https://arxiv.org/abs/2303.10876) adaptation on the same
[OpenDD](https://arxiv.org/abs/2007.08463) pedestrian development roster.
It contains 10,475 windows from 202 recordings at seven locations;
9,823 have all twelve future labels and 53 have none. The existing
8-observation/12-prediction protocol at source stride 12 is unchanged.
The units are publisher coordinates. The corpora differ in motion and
sampling as well as label coverage, so this is not a causal estimate
of the effect of missingness.

The three spline variants have the same architecture without peers.
Each fits on six locations and predicts the seventh for three seeds.
The ego EqMotion control uses the same fitting locations, exact
site-balanced observed-row sampling, robust loss, optimizer, and 800
updates. I verified the final sampling RNG against its checkpoints.
The models are not parameter-matched, and EqMotion is a fixed-head
adaptation rather than a reproduction of its published benchmark.

| Forecast | Complete ADE | Available endpoint FDE | Conservative gain vs CV, pp [95% CI] |
| --- | ---: | ---: | ---: |
| CV | 0.556 | 1.227 | Reference |
| OLS4 | 0.620 | 1.319 | -10.028 [-12.952, -6.496] |
| Ego EqMotion | 0.561 | 1.248 | -0.460 [-2.718, +1.647] |
| Centered OLS4 spline | 0.538 | 1.186 | +4.858 [+0.759, +10.150] |
| Centered CV spline | 0.550 | 1.211 | +4.375 [+0.190, +9.923] |
| Comoving CV spline (registered primary) | 0.545 | 1.214 | +1.136 [+0.004, +3.028] |

The primary observed gain is +2.143 pp [+1.025, +3.768] over CV.
The conservative interval barely excludes zero and only four of
seven locations have positive seed-averaged conservative gains.
The smallest location contributes +6.509 pp with only 12 complete
windows; an exploratory leave-one-location summary excluding it is
+0.241 pp [-0.101, +0.610], not a substitute for the primary roster.
The direct contrast against ego EqMotion is +0.111 pp [-1.863, +2.186].
At matched architecture and loss, the comoving transform plus its
observation scale loses 5.598 pp [-8.607, -2.737] versus centered CV.
I report the centered results as controls, not held-selected winners.

## Original Risk and Fixed Interventions

Unconditional comoving forecasts pass 11/606 original recording-seed
risk checks; worst defined signed easy degradation is +298.44%.
This raw forecast is not suitable for replacing CV.

I next hold all 1,457 recipients, query times and peak displacements
fixed. Each new forecast supplies only its direction at a frozen slot.
The two-choice capacity is the result of allowing the original or new
shape per slot with outcomes known; it is not an online policy.

| Fixed-slot forecast | Original risk passes | Positive against both references and risk pass | Two-choice hindsight capacity |
| --- | ---: | ---: | ---: |
| Original | 112 | 77 | Not applicable |
| Existing past-prefix choice | 124 | 90 | Not applicable |
| Ego EqMotion | 117 | 85 | 165 |
| Centered OLS4 spline | 110 | 80 | 144 |
| Centered CV spline | 115 | 84 | 144 |
| Comoving CV spline | 92 | 71 | 170 |

The comoving spline loses 0.099 conservative pp [-0.280, +0.008]
versus the original shape in the direct paired fixed-slot comparison.
All arms have 334 undefined easy-reference views. The 606
recording-seed views repeat the same 202 recordings; the 3,000
bootstrap replicates resample locations after averaging seeds.

**Decision:** keep the existing deployment. The positive full-roster
forecast signal and 170 hindsight passes warrant using the spline as
one option in a later risk-aware, source-separated choice test, not as
an unconditional replacement. The next experiment should compare a
causal rule against the original and existing prefix rule at the same
budget, using fitting predictions that exclude the held source.

**Provenance:** 63 fresh ARM CPU fits, 50,400 updates, four Torch
threads and no workers. Training and scoring took 70 seconds with
about 0.66 GiB maximum RSS. Existing cohorts, EqMotion checkpoints,
and frozen intervention slots are `cached_verified`. A zero-update
replay reproduced all forecast paths and scalar results. DroneCrowd
calibration and independent confirmation are `not_run`. No original
risk rule, deployment, Stage5C, or SMC change.

[Protocol](protocol.md) | [Direct readout](readout.json) |
[Fixed slots](slot_readout.json) |
[Manuscript](../evidence_manuscript_v149/README.md)
