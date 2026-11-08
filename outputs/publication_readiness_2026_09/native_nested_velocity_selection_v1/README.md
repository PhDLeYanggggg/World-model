# Can a causal selector find the useful neural edits?

The previous loss experiment showed hindsight headroom, not a policy.
I froze its masked-symmetric velocity forecast and fitted two rankers:
one for lower utility, and one that also estimates upper positive harm.
They see the same past ego and peer features and causal proposal
geometry. The second ranker subtracts its harm estimate from its gain
estimate; its threshold is not calibrated.

I used nested source exclusion. When scoring a held source, neither
the forecast nor the selector sees that source's training outcomes.
Each selector-training row is also predicted by a donor forecaster
that excludes its own source. Three seeds and 36 unordered donor
pairs per seed give 108 fresh donor fits; the two selector arms give
54 fresh fits. The 27 outer forecast fits are verified cache. The
9 sources contain 69,746 windows and 1,176 eligible queries. See the
[registered protocol](protocol.md).

## One edit per query

Each matched arm makes one edit on the same 1,176 queries per seed.
Numbers are full-grid lower gain against OLS4, in CV-error-normalized
points, with 3,000 source resamples after seed averaging. Every missing
step pays forecast disagreement. The oracle can see outcomes.

| Ranker | Lower Gain [95% CI] | Positive Edits out of 3,528 |
|---|---:|---:|
| No edit (OLS4) | 0 | N/A |
| Gain only | -0.1814 [-0.3546, -0.0297] | 1,468 |
| Gain and harm | -0.1592 [-0.3386, +0.0013] | 1,497 |
| Smallest trajectory change | -0.0755 [-0.1852, +0.0148] | 1,469 |
| Random | -0.6107 [-0.9641, -0.2991] | 1,285 |
| Hindsight | +1.0471 [+0.6094, +1.5010] | 3,315 |

The learned ranker does not beat no edit. Its separate lower score
is less negative than random, but a direct pairwise lower contrast
does not establish superiority over random or minimum displacement.
Subtracting two separate lower scores is not the lower score of
their paired difference.

With the strict predeclared positive-score option the harm-assisted
ranker makes 1,445 edits across seeds. Its lower gain is -0.0229
[-0.0652, +0.0071]. It reduces observed easy harm, but one source
lacks a denominator in each seed. At matched counts the worst
supported known easy harm is 2.304% in one seed, relative to OLS4
edits with the original CV easy-error denominator. This diagnostic
does not certify safety.

## Is OLS4 the reason for the failure?

I kept the selected trajectories fixed and applied them on top of
CV instead. The gain-plus-harm decisions have +0.1558 points on
observed grid steps, but -0.1707 [-0.4737, +0.0772] after a 0.3265
missing-step debit. The decisions still do not establish gain. At
the same count, a separate CV-ranked hindsight oracle reaches
+1.6843 [+0.8400, +2.7508]. The OLS4-ranked oracle transfers only
+0.4674 [+0.1467, +0.8159] to CV. The reference affects which
candidate is useful. [Fixed-action diagnostic](floor_protocol.md)

The most direct follow-up is to train the ranker for the actual CV
fallback, holding candidate forecasts and edit count fixed. A
query-level ranking loss is a distinct hypothesis: choosing the best
candidate within a query may matter more than regressing absolute
per-row costs. Two-stage regression deferral is established
([Mao, Mohri and Zhong, ICML 2024](https://proceedings.mlr.press/v235/mao24d.html));
the present rankers are not a replication of its surrogate, and
using two stages by itself is not a contribution.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_nested_velocity_selection.py train
.venv-pytorch/bin/python scripts/run_m3w_nested_velocity_selection.py score
.venv-pytorch/bin/python scripts/run_m3w_nested_velocity_selection.py replay
.venv-pytorch/bin/python scripts/report_m3w_nested_velocity_floor.py
```

Use `train --resume` with existing checkpoints. CPU training uses
four threads, no loader subprocesses, and atomic checkpoints every
100 updates. Fourteen targeted tests passed; [scores](analysis.json)
and the [fallback diagnostic](floor_diagnostic.json) replay exactly.
The weights and individual forecasts stay private.

This is a fitting-source experiment in image coordinates over
8 observed and 12 predicted annotation steps. It is not independent
evidence or a metric/seconds claim. The DroneCrowd 15/30 split and
2% rule remain unchanged; no new calibration or confirmation outcomes
were opened. No deployment, Stage5C or SMC.
