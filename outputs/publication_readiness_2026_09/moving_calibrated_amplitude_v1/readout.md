# Learned Amplitude At Fixed Intervention Count

**Decision:** Keep the original evaluation rules and the existing model
policy. Fixed-quarter amplitude is the development control for this
comparison; none of the new variants clears the complete safety gate.
Registration: `acf057a7`, before fitting or scoring.

## Question And Design

Can a model learn how much to move toward a neural forecast, instead of
applying the same one-quarter adjustment to every selected case? All
arms retain 6,952 switches (1.846%) and identical forecast candidates.
Only the interpolation coefficient changes. The 102 views come from 34
previously exposed development recordings and three forecast seeds,
not 102 independent scenes. In each of the 42 fits, seven sources train
the amplitude predictor, six select the policy, and one is excluded
from both. The 32-wide network predicts utility and positive all/easy
harm for five amplitudes. No future coordinates are inputs.

## Results

| Arm | Original Passes / 102 | Total Risk >2% | Signed Easy >2% | Missing Easy Reference |
|---|---:|---:|---:|---:|
| Original direct | 2 | 75 | 8 | 35 |
| Fixed quarter | 15 | 39 | 0 | 35 |
| Donor-calibrated uniform | 15 | 39 | 0 | 35 |
| Donor-calibrated adaptive | 14 | 39 | 0 | 35 |
| Matched query budget (diagnostic) | 19 | 39 | 0 | 35 |

| Paired Contrast | Completion Lower (pp) | Source Bootstrap 95% Interval (pp) |
|---|---:|---:|
| Fixed quarter vs causal constant velocity | +0.05676 | [+0.01788, +0.09511] |
| Uniform vs fixed quarter | 0 | [0, 0] |
| Adaptive vs fixed quarter (primary) | +0.00303 | [0, +0.00909] |
| Matched query budget vs fixed quarter | -0.00087 | [-0.00316, +0.00153] |

All 42 uniform decisions and 41 of 42 adaptive decisions fell back to
quarter. The remaining adaptive fold changes just 12 switches. Its
interval touches zero and it loses a pass on aggregate, so it does not
justify replacing the control. The matched diagnostic reallocates
coefficients for 2,219 query pairs, keeping both the count and sum
of coefficients in each query. It is not matched on physical distance.
Four more passes are not evidence of added forecast utility.

## What The Model Did Not Fix

The aggregate total-risk violations and missing references are unchanged
among the reduced-amplitude arms. Shrinkage helps compared with the
original direct model, but reallocating positive amplitudes over the
same roster cannot create evidence for an absent reference. The
next model-side question is whether reducing observed positive harm by
changing which cases receive an intervention can preserve matched-count
utility on excluded sources. No identity exception or threshold change
is used, and old failures remain failures.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_calibrated_amplitude.py fit
.venv-pytorch/bin/python scripts/run_m3w_calibrated_amplitude.py score
.venv-pytorch/bin/python scripts/run_m3w_calibrated_amplitude.py replay
```

Native arm64 PyTorch: 42 fresh fits, 220 updates each, four CPU threads;
fit 40.9 s, score and replay about 10 s each. The old forecast banks
are verified cached inputs. Both controls reproduce exactly and the
[replay receipt](replay_execution.json) binds the same
[analysis](analysis.json) as the initial score. Checkpoints and
per-row caches stay local. The targeted regression suite passed 31 tests.

This is 8 observed / 12 predicted steps in uncalibrated image
coordinates. Completion bounds are not full-grid ADE for partial
tracks. No independent confirmation, physical time or scale claim,
true 3D, foundation claim, deployment change or submission-readiness
claim follows. DroneCrowd outcomes were not opened. Stage5C and SMC
remain off.
