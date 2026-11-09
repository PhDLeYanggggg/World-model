# AV2: Edit Timing Changes the Candidate Ceiling

The [relative-margin experiment](../av2_relative_margins_v2/README.md)
found 48 of 1,020 two-slot views with a qualifying alternative
to the fixed independent choice. I kept the predictors, edit
count, two peak displacements, and original risk rule fixed,
and asked whether alternative edit timings change that ceiling.

For each CV-relative offset sequence I apply either a decreasing
or an increasing linear weight from 0.25 to 1, then renormalize
it to its original peak. Directions at each step are unchanged;
the displacement profile is not. The original, early-weighted
and late-weighted banks have the same eligible actors and exact
number of options. Their union permits mixed timings. It has
nine times as many two-slot assignments; it is not a
same-compute improvement. This is a forecast edit, not a claim
about physically feasible vehicle controls.

The distinction is useful alongside
[CoverNet](https://openaccess.thecvf.com/content_CVPR_2020/html/Phan-Minh_CoverNet_Multimodal_Behavior_Prediction_Using_Trajectory_Sets_CVPR_2020_paper.html):
that work designs sets for state coverage and classifies possible
motions; here I ask how many exact-budget *changes from CV* can
beat both motion floors and a matched independent path under
the unchanged harm constraints. The frozen predictor bank
includes the existing local 8/12
[EqMotion](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.html)
control. It is not a claim about the publisher leaderboard.

## Results

The 1,020 seed views correspond to 340 scenarios in six
development cities. Each arm uses both motion-floor checks
and the three original 2% risk tests, with missing steps
charged exactly as before. Candidate-menu hashes, frozen
additive selections and calibration thresholds are saved
before held scoring. I replay the old fixed choice exactly
and reproduce the four prior corrected capacity counts.

| Bank | Options | Any eligible reference | Any eligible pair | Pair vs original fixed reference | Frozen scorer: eligible / with a qualifying alternative |
|---|---:|---:|---:|---:|---:|
| Original | 411,251 | 382 | 344 | 48 | 63 / 48 |
| Early only | 411,251 | 318 | 286 | 49 | 47 / 36 |
| Late only | 411,251 | 394 | 349 | 31 | 68 / 48 |
| Union | 3,701,259 | 419 | 409 | 61 | 50 / 42 |

The capacity columns use held outcomes to count witnesses,
not to select a policy. Against the unchanged independent
reference, the union adds 13 views with a lawful joint
improvement. The paired city-balanced change is **+1.13 pp
[+0.38, +1.88]**; on complete-label views the count is
43 vs 33. Allowing both paths to vary gives +7.69 pp
[+4.66, +12.14]. The matched counts are smaller: changing
the reference with hindsight is not an inference rule.
Intervals use 3,000 city resamples after averaging seeds
within scenarios.

The unchanged additive scorer, forced onto the union,
*loses* 1.20 pp of eligibility [-2.09, -0.33], and its
conservative trajectory gain relative to the original
forced scorer is -2.40 pp [-3.24, -1.58]. Late alone
adds five eligible choices, but its paired interval includes
zero, and its only calibrated held release fails. Early
releases twice and neither passes. Neither original nor
union releases in held cities. Palo Alto has 26 possible
pairs in the union versus 16 originally, but zero eligible
causal selections with any arm.

Timing is thus useful for candidate supply. Simply exposing
an existing scorer to more options is not enough. A matched
refit should test whether the extra options can be ranked
from history, with the original candidate bank as the
training and decoding control.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_temporal_menu.py freeze
.venv-pytorch/bin/python scripts/run_m3w_av2_temporal_menu.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2*.py \
    tests/test_m3w_interaction_frontier.py tests/test_m3w_slate_relative_gain.py
```

Native arm64, four Torch threads, one process. Freeze 14.22 s
(0.67 GB maximum RSS); score 8.44 s; 67 tests passed. The
18 frozen fold receipts support a restart and are checked
against the existing city-excluded assets. No new model
training, reserved outcomes, deployment or manuscript change.

[Registration](registration.json) · [Freeze](freeze.json) · [Metrics](summary.json)
