# AV2: Supported Gain Helps Retrieval; the Fixed Control Limits Release

The [signed-gain selector](../av2_pair_refit_v1/README.md) multiplied
qualification probability by an unconditional gain estimate clipped
at zero. A candidate could score zero even if its gain when qualified
was useful. I changed the target and kept the candidate set fixed.

## Same-Budget Refit

Both alternatives use the same 128 publisher-training scenarios,
8 observation steps and 12 prediction steps, native units, frozen
candidate paths and 459 edit peaks. Each fold fits on four cities,
calibrates on a fifth and scores on a sixth. The three seeds and
the 320-update budget are unchanged. The two architectures share
the parent MLP, initialization and minibatch schedule:

- **Conditional:** regress the nonnegative gain only on qualifying
  candidates; multiply the prediction by estimated qualification
  probability. The log-odds adjustment removes the fixed positive
  weight of five from the weighted classification score. This does
  not guarantee probability calibration.
- **Censored:** regress gain on qualifying candidates and zero
  otherwise. This estimates the supported gain directly.

Both use squared loss for the gain target and keep the auxiliary
qualification and support losses. Gain output is bounded by the
causal pair-relative displacement from the common reference.
Neither method applies the old sign test. Calibration still requires
original 2% risk checks and positive gain over CV and OLS4 for both
arms, plus pair lift and individual accuracy noninferiority for
joint selection versus additive. Any choice between the two methods
is made on calibration only. An empty policy returns CV.

## Retrieved Options Are Not Reliable Forecasts

| Held metric | Signed product | Conditional | Censored |
|---|---:|---:|---:|
| Qualifying candidate in top five, out of 47 possible views | 1 | 18 | 14 |
| Qualifying candidate selected, out of 47 | 0 | 3 | 3 |
| Known ADE | 0.26371 | 0.26350 | 0.26399 |
| Available FDE | 0.59152 | 0.59662 | 0.59763 |
| Conservative trajectory gain vs CV (pp) | -2.79 | -3.73 | -4.21 |
| Hard gain vs CV (%) | -2.14 | -2.13 | -2.20 |
| Risk passes / 285 edited views | 6 | 10 | 4 |
| Undefined edited-easy ratio | 247 | 243 | 240 |
| Signed easy degradation over 2% | 27 | 26 | 35 |

CV remains the strongest tested reference (ADE 0.25617, FDE
0.57731); frozen joint scores ADE 0.26084 and FDE 0.58323.
The signed-product and frozen controls replay exactly. No new model
has a positive head-to-head full-grid pair gain against frozen joint:

| Direct contrast, native units | All scenarios, gain [95% CI] | 59 complete scenarios, gain [95% CI] |
|---|---:|---:|
| Conditional minus frozen | -0.01443 [-0.02850, +0.00069] | -0.01879 [-0.03516, -0.00424] |
| Censored minus frozen | -0.01496 [-0.02881, -0.00038] | -0.01917 [-0.03500, -0.00492] |

The intervals use 3,000 paired city bootstrap draws after averaging
seeds within scenario. The failure persists with complete labels.
The new targets recover some options that the old score suppresses,
but ranking and forced accuracy are not good enough. Both arms fail
all **348 nonempty calibration breakpoints** (174 each). Three
conditional and five censored calibration choices qualify against
the additive path when the reference is not itself required to
pass; none also has a qualifying additive control. All 18 folds
return CV. This is not a release or an independent validation.

## Why the Same Fixed Control Is a Poor Next Training Target

I enumerated every legal two-slot assignment after the forecasts
were frozen. The outcome-based counts below are upper bounds,
not candidate inputs, threshold choices or new policy performance.

| Held opportunity | Views / 174 |
|---|---:|
| Joint candidate qualifies against the existing additive reference | 47 |
| Existing additive has enough support and passes risk | 5 |
| Existing additive beats CV and OLS4 with conservative utility | 71 |
| Existing additive passes both | 3 |
| Joint candidate and existing additive both qualify | 2 |
| Any legal action beats CV and OLS4 and passes all risk checks | 66 |
| Any two such actions exhibit positive pair lift without individual accuracy loss | 57 |

The two opportunities with the fixed reference come from one
scenario. Neither occurs in the 81 fully labelled two-slot
seed views; the unrestricted-reference ceiling finds 36 there.
The fixed control lacks an edited-easy denominator in 148 of
174 two-slot views. The calibration totals above the line are
numerically identical because the same city set rotates through
the two roles; they are not separate independent samples.

The **2-to-57** contrast shows that the original rule and edit
menu do not by themselves rule out joint value. The frozen
additive *decision* removes nearly all opportunities before the
joint policy can demonstrate it. The 57 uses hindsight to choose
both arms and is not a result for a causal additive selector.
The next experiment should learn an additive control that
prioritizes dual utility and edited-easy support from donor
cities, then compare it with the joint policy using exactly
the same menu, budget, risks and calibration. A larger
development set and a matched public forecaster remain needed;
the existing [8/12 EqMotion adaptation](../opendd_comoving_eqmotion_v1/README.md)
is a more comparable starting point than importing a pretrained
[QCNet](https://github.com/ZikangZhou/QCNet) configuration using
50 observed and 60 predicted steps.

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_supported_gain.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_supported_gain.py score
.venv-pytorch/bin/python scripts/audit_m3w_av2_paired_release_capacity.py
```

The Mac completed 36 arm64 CPU fits in 24.13 s and scoring in
4.68 s, under 727 MB RSS. Four Torch threads, zero data workers.
Checkpoints every 40 updates preserve optimizer and sampler state;
the same command resumes. The heartbeat records phase and PID.
Twenty-six focused tests passed. No CREATE submission, official
AV2 validation/test access, reserved confirmation access, Stage5C,
SMC, deployment or manuscript change.
[Registration](registration.json), [freeze](freeze.json),
[metrics](summary.json), [paired-release ceiling](paired_release_capacity.json).
