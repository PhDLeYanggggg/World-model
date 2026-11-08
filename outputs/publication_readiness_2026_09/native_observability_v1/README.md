# Better label-support estimates do not yet improve trajectory decisions

I kept the candidate forecasts, quarter amplitude, two changes per
query and eight observed steps from the recipient-ranking experiment.
I added twelve past box features, including size changes, center
jitter, image-edge clearance, current box overlap and the past
visibility of nearby tracks. These are proxies from released boxes,
not detector confidence, verified online tracking or physical
occlusion labels. Target histories already have eight consecutive
observations. No extra history or future mask enters model inputs.

The feature builder matched all 69,746 target windows across 24
admitted fitting recordings to the previous roster and head-box
centers. First-frame dimensions were bound to the annotation hash.
71 prefix-truncation checks reproduced features exactly. I fitted
108 small Torch networks across three seeds and nine held sources,
crossing direct and factorized utility with and without box features.
All arms use the same candidate, model size and training draws. In the
factorized arm, predicted available-step gain is combined with the
known disagreement cost of an unavailable step.

## Availability Prediction

Macro averages over the 27 held-source/seed cells; positive means an
unavailable future label, not verified physical invisibility.

| Support model | Brier (lower is better) | AUROC | AUPRC |
| --- | ---: | ---: | ---: |
| Donor prevalence by lead | 0.1907 | 0.696 | 0.386 |
| Direct, motion only | 0.1584 | 0.714 | 0.468 |
| Direct, box quality | 0.0781 | 0.914 | 0.832 |
| Factorized, motion only | 0.1603 | 0.718 | 0.478 |
| Factorized, box quality | 0.0782 | 0.913 | 0.834 |

The factorized box model lowers Brier by 0.0821 [0.0603, 0.1129]
relative to factorized motion. This paired 95% interval resamples
the nine sources after seed averaging, and all nine improve. The
mean missing fraction is 0.262. The prevalence comparator varies
with forecast lead, so its discrimination need not be 0.5.

## Trajectory Decisions

All active arms make 2,352 changes per seed over 1,176 queries.
The units below are gain points on the observed CV error
denominator, not percent ADE reduction. Use 3,000 paired
source resamples; evaluate direct policy-to-policy bounds.

| Contrast | Observed gain | Full-grid lower gain [95% CI] |
| --- | ---: | ---: |
| Factorized quality vs OLS4 | +0.1865 | +0.0462 [-0.0984, +0.1905] |
| Factorized quality vs minimum change | +0.1722 | -0.0018 [-0.1466, +0.1558] |
| Factorized quality vs factorized motion | +0.0800 | -0.3011 [-0.5609, -0.0726] |
| Direct quality vs direct motion | +0.0519 | -0.0595 [-0.1136, -0.0108] |
| Direct quality vs OLS4 | +0.1059 | +0.0282 [-0.0434, +0.0778] |

Direct lower bounds are not differences of two lower bounds against
OLS4. Negative lower bounds do not prove that the actual unobserved
outcomes were worse. Factorized quality gives available ADE/FDE
91.438/234.183 pixels, versus 91.597/234.452 for OLS4. Hard-window
contribution is -0.0154 points. Worst supported known-easy positive
harm is 0.32%-0.40% against OLS4 but 260.91%-261.32% against CV.
One source has no easy denominator. The original risk rule does not
pass; no policy or gate is promoted.

## What Changed In The Decisions?

I froze the conditional gain head and substituted only the support
probability. The mean selected missing-step fraction (query-weighted
and source-averaged) falls from 25.62% with factorized motion to 10.29%
with factorized box quality. Holding the quality gain head fixed,
its OLS4-relative lower gain is +0.0462 with box support, -0.2806
with motion support and -0.8204 with donor prevalence. These are
separate comparisons with OLS4, not a paired gain claim.

Giving the frozen model the true future mask **only in a hindsight
diagnostic** raises the OLS4-relative lower score to +0.1603
[+0.0529, +0.2815]. It is unavailable at inference and its direct
bound versus predicted support is not positive. That experiment
separates uncertainty about the label mask from error in the
conditional gain head. The new quality signal is useful, but the
next test must improve conditional gain ranking at matched support
and counts, rather than merely predicting more available labels.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_observability.py train
.venv-pytorch/bin/python scripts/run_m3w_observability.py score
.venv-pytorch/bin/python scripts/run_m3w_observability.py replay
.venv-pytorch/bin/python scripts/report_m3w_observability_swap.py
```

Local native arm64 CPU: four compute threads, zero loader workers,
500 updates per fit, checkpoint every 100. Training took 34.0 s;
`train --resume` continues interrupted fits. Nine targeted tests
passed; scoring and the support swap each replayed exactly.
[Protocol](protocol.md) | [Results](analysis.json) |
[Frozen-head swap diagnostic](support_swap_diagnostic.json).

Fitting-source evidence only. No external calibration, independent
confirmation, threshold change, deployment, Stage5C or SMC. Horizons
are annotation steps and coordinates are image pixels.
