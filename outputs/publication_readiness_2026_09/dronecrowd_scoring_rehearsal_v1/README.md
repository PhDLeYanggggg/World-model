# External Scorer Rehearsal on Exposed Development

I used the frozen predictions and decisions from all 34 native
development recordings to exercise the source-group score reducer.
The run covers 102 views, three forecast seeds and 14 source groups.
It uses the original partial-label recording scorer. Query-weighted
masses are merged inside each source/seed cell; seeds are averaged
before the 3,000-draw paired source bootstrap. Missing recording
views are refused and nonestimable cells are not silently dropped.
This is not a DroneCrowd outcome run.

## Accuracy and the Causal Reference

The original risk reference is causal constant velocity (CV).
OLS4 was selected on the producer fitting data as an accuracy
comparator. It is not selected again here or declared strongest
on DroneCrowd. The JSON retains all six causal forecast comparisons
and the three archived neural predictor comparisons on the same rows.

| Frozen comparison | Known-label change (pp) | Completion-lower change (pp) | 95% interval for lower change |
|---|---:|---:|---:|
| Direct policy vs CV | +0.107 | +0.097 | [-0.036, +0.219] |
| Direct policy vs fitting-selected OLS4 | +4.066 | -9.719 | [-21.915, -0.403] |

Both comparisons use the original CV error mass as denominator.
The lower changes subtract the disagreement radius on wholly
unobserved rows. Their pair-specific penalties do not compose;
the negative lower contrast against OLS4 does not prove worse
true complete-future error. The existing direct policy still
passes only 2/102 recording views and is not a safe upgrade.

## Why External Reference Cuts Need a Freeze

The original protocol uses a pair of easy/hard cuts from the
source-excluded producer fit corresponding to each development
source. An external recording has no such producer-to-source
mapping. To measure the practical effect, I applied each existing
pair uniformly as a **counterfactual on development only**. The
choices, errors, 2% limits and all-roster utility did not change.

| Reference definition | Easy / hard (pixels) | Supported views | Undefined easy | Signed easy >2% | Total risk >2% |
|---|---|---:|---:|---:|---:|
| Original source-specific | two fitting-only pairs | 2/102 | 35 | 8 | 75 |
| Existing pair 0, uniform | 9.546 / 29.900 | 1/102 | 36 | 8 | 75 |
| Existing pair 1, uniform | 15.404 / 49.249 | 3/102 | 34 | 1 | 75 |

The sensitivity exercise neither replaces the original result
nor selects the more favorable pair. A future external protocol
must freeze the cutoff definition and the controller before
confirmation; risk stays at 2%. Zero switches and undefined
references remain distinct from a pass.

```sh
.venv-pytorch/bin/python scripts/run_m3w_external_score_rehearsal.py score
.venv-pytorch/bin/python scripts/run_m3w_external_score_rehearsal.py replay
```

Fresh local score and replay take about 16 seconds each and agree
exactly. No new training, external inference, confirmation label
access, role assignment, deployment, Stage5C or SMC took place.
Pixels and annotation steps are not metric distances or seconds.
