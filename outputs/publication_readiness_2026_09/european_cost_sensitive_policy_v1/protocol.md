# Cost-Sensitive Primal Repair, Fitting Only

## Motivation and Fixed Scope

The corrected TRAIN diagnostic finds a positive safe label-aware selection in
146/148 recording/context views (74 unique source-qualified recordings).
One recording, locality082/51, lacks an admissible easy anchor in both forecast
contexts. It stays in training/evaluation; this experiment cannot certify the
complete roster with the current fixed action set. A fitting improvement on
other records will be reported as such, not used to hide that obstruction.

The cross-record model captures only23.46% of eligible positive benefit. In
371/444 repeated recording/head-seed views its own choices contain a safe useful
deletion-only oracle, while its actual policy meets budgets in160/444. This
motivates checking the decision surrogate before another architecture search.
The oracle is not a causal policy or evidence that its action set is learnable.

Train18 new neural intervention heads: six fixed contexts x seeds17/29/43,
2,000updates each. Match the frozen exact_gradient comparator in input rows,
features, initialization, architecture385->32->1, optimizer/dual settings,
source/record weighting, temperature schedule and final logit>0 threshold.
Retain origin as a secondary control. No new forecast network is trained.
All sources and recordings remain. No best checkpoint/seed/threshold selection.
Evaluate only fitting recordings. New inner-held scores, independent selection,
risk calibration, confirmation and deployment are not run in this experiment.

## One Changed Mechanism

Keep the same per-row Lagrangian coefficient
`c = -gain_lower + dual_all*risk_all/scale_all + dual_easy*risk_easy/scale_easy`.
The previous primal surrogate is `c * STE(sigmoid(logit/temperature))`.
Replace only its primal optimization with the cost-sensitive logistic loss
`abs(c) * softplus(sign(c) * logit/temperature)`. Coefficients/duals are detached
targets, not inference inputs. The dual player still observes exact hard-decision
recording risk, learning rate1 and cap100. AdamW lr.001, weight_decay.0001,
gradient cap5, final fixed2000th iterate, temperature1 to.25, no averaging.

This retains the preferred hard action for each fixed coefficient but changes
the smooth optimization problem. An incorrectly confident logit has a vanishing
sigmoid-action derivative; the cost-sensitive logistic derivative remains large.
That analytical property does not establish it caused the real failure or that
the replacement improves fitting. Log derivatives at frozen comparator and new
final states; test the mechanism and measure actual choices separately.

Cost-sensitive reduction is established prior work, not our novelty claim.
[Agarwal et al., ICML2018, Section3.1](https://proceedings.mlr.press/v80/agarwal18a/agarwal18a.pdf)
relate two-action costs to cost-weighted classification. Our inexact neural
gradient update is not their best-response oracle or randomized classifier and
inherits no feasibility/optimality guarantee. Likewise, fitting constraints do
not establish generalization; [Cotter et al., ICML2019](https://proceedings.mlr.press/v97/cotter19b/cotter19b.pdf)
study independent-data constraint enforcement. We do not relabel the exposed
inner-held cohort as such an independent dataset.

## Readout and Limits

Freeze all18 finals, then recompute all72 fitting head-seed views,444 recording
views. Compare exact control hashes/scores, full-policy 2% all/easy positive-harm
ratios and easy degradation, selected known/unknown counts, benefit/harm
retention, and full/count-matched utility completion bounds. Empty matched
risk is not_estimable and utility matching is not a safety gate. These are
TRAIN fitting diagnostics, not test improvement, FDE gain or confidence intervals.
Never infer population or physical safety from empirical training budgets.

Native arm64 .venv-pytorch CPU4/interop1/workers0. Pilot uses40 real updates:
10+10 resumed versus20 direct, exact model/optimizer/dual/RNG equality. Its
engineering result, not predictive performance, admits the full18 fits.
Checkpoint every100updates, heartbeat50, process lock, resume,128MiB checkpoint
cap,10GiB reserve,12h limit. Slow execution does not reduce scope.

Fresh_run: new policy training/readout. Cached_verified: source data, frozen
forecasts/moments/comparators. Not_run: new held evaluation or deployment.
No Stage5C/SMC, metric/seconds, true3D/foundation or submission-ready claim.

```sh
.venv-pytorch/bin/python -m scripts.run_m3w_cost_sensitive_policy register
.venv-pytorch/bin/python -m scripts.run_m3w_cost_sensitive_policy pilot
.venv-pytorch/bin/python -m scripts.run_m3w_cost_sensitive_policy train
.venv-pytorch/bin/python -m scripts.run_m3w_cost_sensitive_policy evaluate
.venv-pytorch/bin/python -m scripts.run_m3w_cost_sensitive_policy check
```

Use --resume only for an interrupted phase. Final model and readout manifests
are immutable. The frozen controls require no duplicate training.
