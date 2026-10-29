# Conditional Calibration Improves Some Risk Views but Loses Useful Selection

I completed the registered72 source-excluded calibration/readout folds using
all18 frozen rational heads. No new neural model or trajectory forecaster was
trained. Source packets/checkpoints are cached_verified; coefficient fitting,
decisions, metrics and accounting are fresh_run. Independent transfer is not_run.

## Fixed Comparison

| Policy | Selected occurrences | Unknown selected | Empty views | Undefined easy risk | Known easy violations | Upper easy violations | Complete support /72 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw rational |27,371|699|3|19|34|51|0|
| Global calibration |21,941|586|3|19|29|47|4|
| Support-only veto |21,332|502|9|25|29|45|0|
| Conditional calibration |5,693|101|25|33|8|18|20|

Complete support means the unchanged finite-completion utility, all/easy
positive-harm and easy-preservation checks all pass. It is not a population
guarantee. Empty/undefined views are not safe successes. Counts are dependent
model-view occurrences, not unique agents or72 independent scenes.

The conditional rule gains complete support in20 views, but loses most utility.
Its selected count is only20.80% of the raw rule. Fewer violations alone would
hide this loss and the increase in undefined denominators.

## Utility at Full and Matched Intervention Counts

Conditional minus control, percentage points of known reference error mass:

| Control | Full-count contrast [95% CI] | Same-count contrast [95% CI] |
|---|---:|---:|
| Raw | -4.646720 [-6.292913,-2.978920] | -0.365796 [-0.704795,-0.108582] |
| Global | -3.540215 [-5.037809,-2.210522] | -0.184432 [-0.344481,-0.046161] |
| Support-only | -4.352740 [-6.157591,-2.509902] | -0.365796 [-0.704795,-0.108582] |

These are paired conservative completion-lower utility contrasts, not ADE/FDE
or historical t50 improvement.3000 locality bootstrap draws average repeated
contexts/seeds within12 exposed TRAIN localities. Coefficients are held fixed;
cross-calibration adds dependence. Intervals are nominal exploration, not
independent or multiplicity-adjusted confirmation. All six utility conditions
and the complete-support condition fail. **No deployment or stage promotion.**

## Why the Result Is Informative

The support-only control shows the loss is not just a lack-of-support veto.
After removing6,039 unsupported occurrences, the conditional penalties remove
another15,639. The known net utility of this second removed cohort was positive:
2.823689pp averaged across localities. At equal counts the conditional rule
still loses, so merely reducing coverage is not the whole explanation.

In23 retained source/seed/cell views with positive observed easy-risk excess,
the applied correction is too small.18 had a positive correction, five had zero.
In11/23, it covered the target's entire raw cell mean residual but not the
subgroup selected after correction. Coarse cell-mean calibration did not survive
its own finer selection. This is descriptive accounting on frozen actions, not
a new causal guarantee or a test-tuned repair.

## Execution and Boundary

- Preregistration/code commit `f4035029`, pushed before real execution.
- Calibration and decision freeze: PID55709 exit0,56.676s, peak RSS2.250GB.
- Final outcome reductions: PID56047 exit0,2.014s, all72 views.
- Full replay: PID56321 exit0; arrays, decisions, old raw predictions/policies,
  all bounds, matched counts and bootstrap summary reproduce.
-26MiB private decision/label cache; no raw data/weights/cache committed.
- CREATE assets reachable:24 original TRAIN packets. Scheduler query timed out;
  queue status is unobserved, not failed. No new remote job submitted.

All18 neural models, upstream producers,2% budget, outer data roles and
deployment remain unchanged. Obs8/pred12, raw stride12, detector-silver,
pixel/dataset-local only. No metric, seconds, human-gold, physical safety,
true3D, foundation or submission-ready claim. Stage5C and SMC remain off.

See [failure accounting](failure_analysis.md), [machine summary](summary.json),
[frozen protocol](protocol.md), [replay guide](verification.md).
