# Seed Disagreement Improves Cost Fit, Not Safe Intervention

I kept the trajectory bank and each query's switch count fixed and tested
whether between-seed disagreement helps identify worthwhile switches. All
125,540 exposed A rows (1,613 queries, 34 recordings, 14 sources) are retained
for each of three actual EqMotion producer seeds. These are 14 resampling units,
not 376,620 independent samples. All three forecasters excluded A from training.

| Full 12-step contrast: dispersion minus single seed | Mean pp | 95% exploratory source interval |
|---|---:|---:|
| Known-label utility | +0.0494 | [-0.0405, +0.1457] |
| Paired completion-lower utility | -3.5347 | [-8.9815, -0.3555] |
| Known easy positive-harm increase | -1.7220 | [-6.0308, +1.0687] |

**The registered component fails.** Both arms switch 3,225 rows per seed.
They disagree on 2,064, 2,120 and 2,164 row decisions for seeds 17, 29 and 43.
The known gain is small and uncertain; the completion-lower contrasts are
-5.372, -3.455 and -1.777 pp, respectively. Easy harm is not reliably reduced.
On equal source/seed means, gain MSE falls from 6,328.33 to 5,697.81 px squared
and harm MSE from 260.29 to 242.04. These are held cost regressions, not policy
improvement or calibrated uncertainty estimates.

## Why The Model Does Not Progress

- Relative to CV, using raw EqMotion throughout has +18.19% known utility
  with 64.23% known easy positive harm. The equal-count control and dispersion
  policies give +1.52% and +1.57% known utility; easy harm is 6.20% and 4.47%.
  In the dispersion arm, 40 of 102 repeated recording views exceed 2% known
  easy positive harm, and 14 exceed 2% known all positive harm. Neither is a
  full original risk evaluation or a safety certificate.
- Each seed has 2,824 wholly unknown rows. The difference between the known
  contrast and its completion lower is 3.5842 pp. Source 087 has the largest
  source-average bound subtraction (37.1290 pp). This subtraction measures
  permitted forecast disagreement on unobserved rows, not observed harm,
  nor evidence that missing labels alone caused all model errors.
- Better average cost fit did not produce better choices in the part of the
  distribution that limits the conservative result. I do not retain the extra
  three-predictor compute as a deployment requirement. A next model-side test
  should directly predict the cost of intervening where outcome support is weak,
  rather than add more dispersion features or optimize exposed thresholds.

## Reproduction And Scope

`fresh_run`: 168 boosted regressors (14 held sources x 3 seed banks x 2 arms x
2 targets), then exact checkpoint inference and choice replay in two separate
processes. `cached_verified`: original 4,000-update forecasters and their held
paths. `not_run`: external outcomes, full 2% deployment gate, full three-seed
calibrated controller and independent confirmation. Nothing is deployed.

```sh
.venv-pytorch/bin/python scripts/run_m3w_seed_dispersion_cost.py fit
.venv-pytorch/bin/python scripts/run_m3w_seed_dispersion_cost.py score
.venv-pytorch/bin/python scripts/run_m3w_seed_dispersion_cost.py replay
```

Fit saves each source/seed checkpoint atomically and resumes the same roster;
score and replay each have immutable receipts. PIDs 10474, 10547 and 10564
all exited 0. Fit 33.56 s, score 4.00 s, replay 4.03 s, 1.32 GB peak fit RSS,
20 MB local checkpoint directory. Native arm64, CPU thread limits 4, no
multiprocess data loader. Thirty-four recordings per seed retained; no
undefined source endpoint dropped. No change to existing 2%/support/NA rules,
coordinate interpretation, Stage5C or SMC. This is image-coordinate 8/12
development evidence, not verified seconds, metric or physical safety.

[Protocol](protocol.md) | [All paired source and recording outcomes](analysis.json)
