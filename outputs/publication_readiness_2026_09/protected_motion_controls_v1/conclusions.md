# What Remains After Protecting the Simple Controls?

## Completed Evidence

This registered source-only experiment is complete. It adds **72 neural cost
heads and 84 sampler-matched forests**, and verifies/reuses 12 existing full
Transformer cost heads. These are 156 new control-head fits, not 156 newly trained
world models. All prescribed 3,000-update/128-tree budgets completed; completion
does not assert optimization convergence.

The population is 175,756 past-eligible pedestrian windows from 33 recordings in
four already development-exposed SDD sites. There are 143,918 complete future
paths, 29,039 partial paths and 2,799 paths with no future labels. All remain in
the inference population. Three seeds are used. The primary statistic is the
equal mean of site-relative available-point ADE gains over causal CV, **not**
pooled-window improvement, raw-frame t+50, or the historical Stage37 protocol.

Every cost head excludes its outer site. Transformer training-cost producers
also exclude each training row's site, using the existing pair-excluded bank.
Causal formulas have no learned producer. Preprocessing, complete-label fitting
support and the exact neural sample draws match the registered controls.
All outer choices were saved before outcome readout. No threshold search,
external readout, winner deployment, Stage5C execution or SMC occurred.

Fresh-process inference and aggregate replay are exact. A separate arithmetic
implementation checks 168 fit budgets, 336 policy decisions, 144 matched-count
pairs and 3,360 scene reductions. This is independent arithmetic by the same
agent, **not** an independent research team's reproduction. The scoped test suite
has 61 passing tests; the full legacy repository suite was not rerun.

## Main Result: A Tradeoff, Not Neural Dominance

| Forecast action and cost head | ADE gain over CV | Hard gain | Worst site/seed easy degradation |
|---|---:|---:|---:|
| Full Transformer + neural | 2.437% | 2.292% | 1.067% |
| Damping005 + neural | 3.635% | 4.723% | 2.494% |
| Damping010 + neural | 3.745% | 4.438% | 3.038% |
| Damping020 + neural | 3.706% | 3.652% | 3.737% |
| Full Transformer + forest | 1.302% | 0.539% | 0.176% |
| Damping005 + forest | 1.414% | 1.384% | 0.000% |

All six causal alternatives, including negative and near-zero results, appear in
[results.md](results.md), [strict_controls.csv](strict_controls.csv), and the full
[analysis](analysis.json). The displayed damping005 forest is not a newly selected
deployment winner; the complete family was registered before fitting.

With the **same neural cost learner**, all three damping actions have greater
mean source gain than Transformer. Transformer-minus-damping paired differences
are -1.198, -1.308 and -1.269 percentage points; their nominal four-site bootstrap
intervals are negative. However, all three violate the descriptive 2% worst-site/
seed easy criterion. It would be wrong to call them safely superior.

Transformer/neural retains a useful observed tradeoff: among the fourteen fixed
strict action/head points, it has higher overall gain than the points that also
stay within 2% worst-site/seed easy degradation. This is an observed development
frontier, not a tuned-optimal frontier, certified risk bound or independent proof
that neural forecasts are necessary.

With matched forest protection, Transformer and damping005 are not separated by
the paired interval: Transformer-minus-damping005 is -0.112 pp, CI95
[-0.739, +0.674]. No general forecasting-superiority claim follows.

## Matching Intervention Counts Does Not Resolve Safety

At the same smaller strict count per fold/seed, neural-head damping still yields
greater average gain. For damping005 the comparison is 3.195% versus 2.437% for
Transformer; the paired difference is -0.758 pp, CI95 [-1.325, -0.211]. But its
worst-site/seed easy degradation becomes **3.166%**, versus 1.067% for Transformer.
Damping010/020 matched easy degradation is 3.082%/3.737%.

This rules out selection *volume alone* as an explanation for the gain difference.
It does not establish safe dominance. It also demonstrates why selecting fewer
rows cannot substitute for conditional-risk calibration: some beneficial easy
interventions can disappear before harmful ones. The registered same-count
control is an offline diagnostic, not a query-local deployment policy.

The [matched safety supplement](matched_count_safety.md) gives both arms' safety
statistics, including the zero-intervention acceleration/forest null case. A
0.0% gain from no switching is fallback, not predictive success.

## What Succeeded and What Failed

**Succeeded:** the unfair protected-neural versus unprotected-simple comparison
has been replaced with matched training support, sampling, features, fallback and
all registered action families. Neural cost learning improves over its matched
forest on full Transformer by 1.135 pp, nominal CI95 [0.575, 1.695]. Useful
gain/harm routing is therefore observable on these source folds.

**Not established:** an indispensable neural *forecasting* contribution. Useful
routing also works on simple damping, and the remaining distinction involves
the gain/easy-degradation tradeoff. This is not proof that Transformer is
unhelpful, nor that all neural forecasting is inferior. EqMotion's previously
reported ramp-head comparisons and fixed DUT results use different producers
and must not be substituted for the missing nested full-EqMotion contrast.

**Failed:** the fixed 0.1 predicted-harm/benefit rule is not uniformly easy-safe
across action families. Neural damping violates the empirical ceiling despite
bounded cost outputs. Bounding predicted costs by causal forecast disagreement
does not calibrate their conditional error on selected samples. Constant
acceleration produces negligible or negative protected utility; a trained head
is not automatically useful.

**Unchanged:** the earlier exhaustive joint-opportunity audit found sparse
opportunities, and DUT joint-vs-unary benefit was negligible. This experiment
does not repair joint support, prove a scene/goal or image contribution, or make
M3W a submission-ready world model.

## Integrity, Cost and Claim Limits

- Result origins: new causal and forest fitting / outer evaluation are
  `fresh_run`; twelve existing neural cost heads are `cached_verified` with
  exact input/label/preprocessing/support/hash checks. Nested EqMotion comparison
  is `not_run` here, pending pair-excluded forecasting producers.
- The registered training run spans approximately four hours of wall-clock log
  timestamps. Summed new fitting-loop time is 1,741.19 seconds (29.02 minutes).
  The latter excludes loading, inter-fit overhead and wall-clock interruptions;
  it must not be presented as end-to-end runtime. No size or sample downgrade.
- CPU4/interop1/workers0 in native arm64 Python; checkpoints and resume retained.
  No new CREATE job. No active training/evaluation process remains after checks.
- CIs resample four physical sites, not overlapping windows. They are nominal,
  exploratory, and not multiplicity-adjusted confirmation. Three seeds do not
  create more independent sites. Mean seed errors are not an ensemble predictor.
- Observed-point ADE is not an assumed complete population outcome. Complete
  cases, zero-CV harm, tails, unknown-label selection and full-grid gain bounds
  remain visible. Bounded gain intervals are not absolute physical-safety bounds.
- Histories are past-indexed *offline annotations*. Existing interpolation
  provenance can involve later annotation controls; access auditing alone cannot
  establish real-time sensor causality.
- Eight observed / twelve predicted native annotation steps, annotation pixels.
  No verified seconds, metric scale, true3D or foundation-model claim. No human
  gold label upgrade. DUT is predictively exposed; DroneCrowd remains closed.

## Next Concrete Step

Build the missing **pair-excluded full-EqMotion source producer/control bank**
under the same fixed population and budgets, with a local runtime pilot before
deciding whether CREATE is warranted. This tests whether the present forecasting
limit is specific to the small Transformer, without selecting a new DUT winner.
Do not spend another round merely tuning the 0.1 threshold on exposed outcomes.

Then settle a predictor-agnostic risk-calibration design that includes the simple
motion controls, rather than exempting them from protection. Its certification
and confirmation need sufficiently independent physical sites. DroneCrowd should
not be opened just to rescue a weak source result. Until those prerequisites and
the joint/multimodal gaps are addressed: **not yet submission-ready; no new model
deployment; the ultimate research goal remains unmet.**
