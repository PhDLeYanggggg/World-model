# Can Sparse CV Edits Catch OLS4?

I started every forecast at CV and gave an outcome-aware selector a budget
of up to one changed track per query. Its five alternatives were OLS4,
OLS8, damped OLS4, stabilized turn and stationary. This isolates what the
budget and menu allow, before asking a learned selector to find good edits.

| Hindsight calculation at one edit/query | Full-grid ceiling, CV-normalized pp (95% source bootstrap) | Recordings with positive ceiling |
|---|---:|---:|
| Analytic menu vs. CV, no risk constraint | +2.823 [+1.886, +3.798] | 33/34 |
| Arbitrary perfect edits vs. OLS4, no risk constraint | -28.175 [-51.361, -9.818] | 4/34 |
| Analytic menu vs. OLS4, relaxed original risk constraints | -33.670 [-58.646, -13.928] | 0/34 |

The arbitrary-path bound is deliberately generous: an edit is perfect on
every known step and identical to the comparison reference at every unknown
step. It still cannot give strictly positive full-grid lower gain over OLS4
on 30/34 recordings at a cap of one. In the finite menu, even dropping
selected-reference support and signed easy constraints leaves 0/34 once
the two original positive-harm constraints are accounted for. The separately
scored cap-one utility maximizers give 0/34 joint witnesses under the full
original rule. An undefined ratio is not counted as safe.

The budget curve separates a learning gap from a capacity limit. At caps
of 4, 8, 16 and 32, the risk-relaxed ceiling permits a positive OLS4 contrast
on 3, 10, 18 and 26 recordings, respectively. Even removing the cap leaves
6/34 with a nonpositive risk-relaxed ceiling and a source-average ceiling of
-0.092 pp. A positive ceiling is not a feasible policy or a joint witness.
The next useful model experiment needs both a broader candidate menu and
more than one opportunity per query; better ranking alone cannot rescue
the tested sparse design.

### Why the bound is valid

Relative to fallback CV, the weighted gain for any reference is additive
over edited rows. Choose the best positive action per row, then sum the
top `k` within each query. This is exact for each separate reference when
there is no risk constraint. For the tightened calculation, subtract
nonnegative multiples of the two linear original risk excesses from each
action's gain. Every feasible plan has nonpositive excess, so the resulting
unconstrained top-`k` maximum still bounds its true gain from above. Taking
the minimum over a fixed multiplier grid keeps the bound valid. The bound
is on the conservative full-grid score; it does not estimate absent labels
or rule out more accurate forecasts under all possible future outcomes.

**Scope.** Fresh calculation and exact replay on all 125,540 exposed rows
from 34 recordings and 14 source groups, 3,000 source bootstrap draws. No
model refits, new roles, threshold tuning, or independent data. The 30
DroneCrowd confirmation groups remain sealed. Eight observed and 12 future
native annotation steps in local-image pixels; not seconds, metric, true
3D, physical safety, or a foundation model. Stage5C and SMC remain off.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_cv_anchor_budget.py
.venv-pytorch/bin/python scripts/plot_m3w_cv_anchor_budget.py \
  outputs/publication_readiness_2026_09/moving_cv_anchor_budget_v1/readout.json \
  data/stage_cvpr2027_experiments/moving_cv_anchor_budget_v1/capacity.png
```

[Protocol](protocol.md) | [Readout](readout.json)
