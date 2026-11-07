# Can Common Motion Improve OLS4 Without Changing Relative Paths?

**Finding:** Yes for a fixed uniform mixture against OLS4, on exposed
development data. The learned mixer does not establish a gain over
the fixed controls or CV, and no full-scene model meets the risk rule.

## Mechanism

For each query, let $p_i$ be an OLS4 path in coordinates relative to
agent $i$ at the observation endpoint. The edited path is
$p_i + (\sum_a w_a \bar p_a - \bar p_{\rm OLS4})$, where the three
mean paths come from OLS4, CV and the frozen self-kinematic model.
The weights are nonnegative and sum to one. The shift is identical
for every agent at each step, so every relative vector is unchanged,
even for a missing label. All reconstructed queries pass that check.
This tests a shared displacement mechanism; it does not identify
camera motion or improve pair geometry beyond OLS4.

The learned 24-wide mixer uses past agent motion and causal candidate
means. Seven sources fit, six choose the epoch, and one is excluded
from both. The donor-selected fixed control chooses among four
predeclared mixtures. The uniform mixture is a separate fixed control,
not chosen using a held score. Forty-two fresh fits use 480 updates
each, and the data remain the same 34 recordings across three
forecaster seeds and 14 source groups.

## Results

| Contrast | Observed Gain (pp) | Completion Lower (pp) | 95% Source Interval for Lower (pp) |
|---|---:|---:|---:|
| CV centre vs OLS4 | +7.178 | -0.033 | [-9.139, +7.475] |
| Kinematic centre vs OLS4 | +13.141 | +2.749 | [-9.655, +14.887] |
| Uniform centre vs OLS4 | +11.924 | +8.503 | [+3.672, +13.850] |
| Donor-selected fixed centre vs OLS4 | +12.276 | +4.761 | [-6.232, +14.651] |
| Learned centre vs OLS4 (primary) | +14.057 | +6.893 | [-1.657, +15.507] |
| Learned centre vs donor-selected fixed | +1.781 | -7.422 | [-22.714, +2.169] |
| Learned centre vs CV | +10.099 | -0.390 | [-10.380, +8.142] |
| Uniform centre vs CV | +7.966 | -4.744 | [-19.506, +4.988] |

The denominator is the unchanged CV error mass. A percentage-point
gain here is not the relative percentage decrease of OLS4 ADE. Bounds
account for wholly unknown rows with actual disagreement; available ADE
on partially labeled rows does not identify the full 12-step error.

| Arm | Original Passes / 102 | Total Risk >2% | Signed Easy >2% |
|---|---:|---:|---:|
| OLS4 | 0 | 102 | 90 |
| Uniform centre | 0 | 102 | 91 |
| Donor-selected fixed centre | 0 | 102 | 93 |
| Learned centre | 0 | 99 | 96 |

The full-scene mixtures intervene on all 376,620 repeated target
rows. OLS4 differs from CV on 375,510. They are forecast component
comparisons, not matched-count intervention policies. The learned
model's larger observed gain is not enough to establish a better
completion lower bound than the fixed control. The result supports
separating common displacement error from relative error when
diagnosing a forecast, not replacing the deployed policy.

Joint metrics are an established complementary evaluation
([Weng et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.pdf)).
This constrained comparison is an image-coordinate accuracy ablation;
it does not replicate their multimodal metric or estimate collisions.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_common_motion.py fit
.venv-pytorch/bin/python scripts/run_m3w_common_motion.py score
.venv-pytorch/bin/python scripts/run_m3w_common_motion.py replay
```

Registered in `61790ef0` before fitting. Native arm64 Torch used four
compute threads, zero loader subprocesses and checkpointing every
80 updates. Fit took 24 seconds; score and replay about 12 seconds.
Fourteen targeted tests passed and the [analysis](analysis.json)
replays exactly. No new independent outcome, calibrated coordinate
claim, deployment, Stage5C or SMC.
