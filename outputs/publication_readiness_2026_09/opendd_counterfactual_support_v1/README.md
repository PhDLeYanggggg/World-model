# Does a query replacement erase the easy reference?

I allowed one query replacement per recording, keeping every query's
edit count and peak-displacement multiset fixed. The model sees the
same recipient and shape menu as the previous experiment. It estimates
the gain of replacing the original query and the probability of losing
its easy reference. The direct local model labels loss at the query;
the direct whole model labels loss only if the remaining original
edits also provide no reference. No future outcomes or future queries
enter the features. The policy acts at the first eligible query.

The two direct arms use the same architecture and loss. I trained
42 fits for 25,200 updates, on three seeds and seven held-location
rotations. About 19.4% of the 65,837 donor options lose the query-local
reference, but only 5.7% lose it across the recording. Donor options
repeat across folds and are not independent.

| Policy | Queries replaced | Lost / restored reference | Risk passes | Dual passes |
| --- | ---: | ---: | ---: | ---: |
| Analytic anchor | 0 | 0 / 0 | 112 | 77 |
| Past prefix | Not matched | 0 / 0 | 124 | 90 |
| Direct local | 335 | 17 / 12 | 120 | 92 |
| Direct whole | 365 | 27 / 13 | 112 | 84 |
| Gain only | 377 | 42 / 18 | 106 | 83 |
| Gain only, same queries as direct whole | 365 | 36 / 17 | 112 | 85 |

Direct whole versus local loses **0.010 conservative pp** (95% paired
location interval **[-0.017, -0.002]**). Its advantage versus same-query
gain is +0.005 pp [0.000, +0.015], but it has one fewer dual pass.
Direct local passes two more dual checks than past prefix, but passes
four fewer risk checks and has **-0.036 pp [-0.089, +0.003]** conservative
utility against it, calculated in the follow-up. Worst signed easy
degradation rises from 11.28% for prefix to 15.13% for either head.
No rule improves both utility and the original risk outcome.

## Selection and timing both lose opportunities

Using outcomes to choose one replacement per recording qualifies
**225/606** views, including the original 77. Two locations contribute
143 of the 148 new witnesses. Of the 365 queries the whole model
replaces, 119 actually contain a qualifying option and it picks one in
71. The model misses promising queries and sometimes chooses the wrong
replacement at a promising query. The gap is not evidence for raising
the risk threshold or for a real-world safety claim.

All arms keep 1,457 edits. The 606 views are 202 OpenDD recordings
repeated at three seeds, not independent trials. The 3,000 paired
bootstrap draws average seeds then resample seven locations. No
deployment change, Stage5C, SMC or access to the 30 independent
DroneCrowd groups.

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_counterfactual_support.py --replay
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_counterfactual_support.py
```

Without `--replay` the first command fits or resumes. The second is a
label-aware diagnostic of the frozen experiment. See the
[factorized follow-up](../opendd_factored_counterfactual_support_v1/README.md),
[readout](readout.json), and [capacity audit](opportunity.json).
