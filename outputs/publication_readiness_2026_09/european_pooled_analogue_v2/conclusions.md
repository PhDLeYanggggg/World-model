# Pooled References Repair Coverage, Not Risk Magnitude

Result source: **fresh_run** for all 72 fitting-only retrieval views;
**cached_verified** for inherited neural weights, fixed query selection and the
v1 input audit. No new neural optimizer update, trajectory prediction model,
policy execution, held readout or independent evaluation is claimed.

## Executed Comparison

The only scientific change from the source-local control is the reference bank:
four approved inner-fit sources from the same producer/controller context are
pooled. Each query's source-qualified recording is absent from its donor
statistics, neighbors and labels. Source qualification also changes the fixed
permutation salt; old and new shuffled controls are not identical permutations.
All six contexts, three head seeds and four query sources are retained.

The real four-source pilot passed. Full run PID75317 completed 72 views in
184.717100 seconds, with 6,341,705,728 bytes peak RSS. There were zero neural
updates. The 62,902 repeated query occurrences comprise 61,335 known and 1,567
unknown outcomes; these are not 62,902 independent recordings or people.
Every query now has reference support, versus 52,654/62,902 previously. All
72 primary easy-risk MSE values are defined, versus 24 previously. The unknown
queries remain in the cohort but cannot supply observed-risk labels.

## Full-Roster Results

Errors below are signed-risk MSE after source normalization, averaged equally
over 72 views. They are not trajectory ADE/FDE. Harm masses sum normalized
known-outcome occurrences, not risk ratios or physical injury counts.

| Estimator | All risk MSE | Easy risk MSE | Predicted easy harm | Known false-safe easy occurrences |
|---|---:|---:|---:|---:|
| Frozen original | 0.079458745 | 0.000309373 | 1.375607 | 2825 |
| Estimation continuation | 0.078065818 | 0.000309427 | 2.268195 | 2799 |
| Pooled mean | 0.174918744 | 0.005576636 | 1817.041412 | 1610 |
| Pooled nearest32 | 0.297406867 | 0.004740689 | 697.167173 | 1764 |
| Pooled shuffled32 | 0.216899278 | 0.011333992 | 1836.233479 | 1789 |

Nearest32 easy-risk MSE is 14.99% below the pooled mean and 58.17% below its
shuffled-label control. Exactly 12/18 context/head-seed pairs beat both controls,
meeting the registered descriptive information screen. All three seed averages
have the same easy-risk direction; these are head seeds, not new forecasters.
This supports a limited fitting-information observation, not significance,
calibration, safety, transfer or permission to open a new evaluation role.

The risk-magnitude comparison fails. Nearest32 easy-risk MSE is **15.32 times
the frozen original head's**, and predicted easy harm is **7.30 times** observed
(697.167173 versus 95.560642). Original predictions substantially underestimate
harm; directly averaging reference harm fractions overshoots. The nearest
all-risk MSE is also 70.03% worse than pooled mean. Fewer false-safe rows do not
rescue a model with systematically inflated risk and unmeasured policy utility.

Full-roster AUROC/AP are undefined under the registered no-view-dropping rule:
one all-risk view and 26 easy-risk views lack the necessary event support.
Their absence is not zero performance and is not a reason to drop those views.
Pooled mean and shuffled each have two copied-benefit/new-harm envelope
exceedances; nearest has zero. These are reported, not repaired after readout.

## Descriptive Slices

On the same 24 previously supported views only, pooled nearest easy MSE is
0.002462924 versus previous source-local nearest 0.003420122, a 27.99% reduction.
This matched subset is a diagnostic, not a replacement primary. Its original
head remains better at 0.000876187. Support expansion and magnitude are distinct.

All 12 source aggregates favor nearest over shuffled, but locality020 worsens
substantially against pooled mean: 0.022824479 versus 0.006062968. Two of six
context aggregates worsen against mean. Locality020 is retained. The frozen
380-feature schema includes log(observed scale); it is not fully invariant to
coordinate scale, and unequal reference-source sizes remain a limitation.

## Decision

Do not deploy or blend this estimator, retune neighbors, change the risk budget,
drop locality020 or use held outcomes to select a repair. A passed information
screen cannot substitute for the failed magnitude comparison. The next fitting
experiment must distinguish easy-event mismatch from conditional harm-size
transfer, with donor recording exclusion and matched original predictions
retained. Before implementing a new calibration fit, reconcile the earlier
hurdle, conditional-calibration and selected-risk controls to avoid repeating
an already failed method under a new name. That follow-up is **not_run**.

The current upstream predictors, preprocessing and original selection already
have training exposure. Reference exclusion is not whole-pipeline OOF. No
confirmatory CI is attached to repeated fitting occurrences or one permutation
seed. Obs8/pred12, raw stride12, detector-silver pixel/local coordinates and
2% positive selected harm/reference remain unchanged. No metric, seconds-level,
human-gold, physical-safety, true3D or foundation claim. Deployment is unchanged;
Stage5C and SMC remain off. The research/submission goal is still incomplete.
