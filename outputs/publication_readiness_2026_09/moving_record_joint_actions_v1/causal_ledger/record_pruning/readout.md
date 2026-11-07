# Can Removing Direct Actions Repair Recording Risk?

The previous experiment found six locally useful openings, none
sufficient to make its whole recording pass. Here a deletion
restores the causal floor on an existing direct action. No action
is added, replaced or moved; at most one is deleted per query.
The task is exposed Moving A development, eight observed and
twelve future annotation steps in native image coordinates.

## Label-Aware Capacity

We maximize the retained count under the original 2% and
reference-support rules. The second row also requires nonnegative
paired completion-lower utility versus the direct output. Every
witness is checked using the original scorer, including the
missing-target penalty on deletions.

| Hindsight constraint | Checked optimal feasible | Infeasible with support | No original switch | No selected easy reference | Unresolved |
|---|---:|---:|---:|---:|---:|
| Original risk and support | 32 | 35 | 18 | 17 | 0 |
| Also preserve paired lower utility | 30 | 37 | 18 | 17 | 0 |

The 30 include two existing passes. The maximum-retention witnesses
for the other 28 improve paired lower utility; across the 30 they
delete 335 switches. These are hindsight capacity counts, not
learned passes. An empty direct stream or a stream without selected
easy reference cannot acquire the required support by deletion.
The remaining 37 fail this existing-action and one-removal-per-query
restriction, not every possible intervention.

## Causal Pruning

I fitted 42 heads on donor-excluded direct actions. Each head
predicts paired deletion gain, both risk excesses, signed easy
loss and easy reference from past motion, family identity and
causal cost estimates. No outcome enters the deployed score.
Seven sources fit each head; six separate donors select the
threshold, and one source is held out.

Before checking paired utility, a threshold recovers a pass without
losing an old pass or adding a signed easy violation in **8/42**
calibration cells. Every such threshold has negative aggregate
paired lower utility on its donors. None meets all requirements.
The frozen policy deletes nothing and exactly matches direct:
**6,952 switches, 2/102 complete passes, eight signed easy violations
and 35 undefined easy upper bounds**. The matched deletion control
is empty and cannot identify relative ranking efficacy. Its
mechanically zero contrast is not safety evidence or improvement.

Missing reference support, limits of the fixed action set, and
utility loss on donors are distinct problems. Lower coverage does
not guarantee better subgroup outcomes in selective regression
([Shah et al., ICML 2022](https://proceedings.mlr.press/v162/shah22a.html));
their subgroup estimand differs from the selected-reference
trajectory risk here. The next relevant training target is a
feasible, gain-preserving removal set, not a looser threshold.

## Reproduction

Registration `ef9eddcf` precedes the 42 new pruning fits. Forecasters
are verified cached inputs, not new training. With private parents:

```sh
.venv-pytorch/bin/python scripts/run_m3w_record_pruning.py fit
.venv-pytorch/bin/python scripts/run_m3w_record_pruning.py score
.venv-pytorch/bin/python scripts/run_m3w_record_pruning.py replay
.venv-pytorch/bin/python scripts/audit_m3w_pruning_calibration.py
```

Fitting and oracle verification checkpoint by source/seed and
recording. Replay recomputes policy scores and aggregates checked
solver receipts; it does not re-solve the integer programs.
The donor diagnostic reproduces all thresholds without held
tuning. The 102 views repeat 34 recordings over three forecast
seeds. No deployment or independent confirmation. No metric,
seconds, physical safety, Stage5C, or SMC claim.
