# Calibration After Analytic Bridge Selection

**Result: not a qualifying controller.** This is a development experiment
on seven previously used OpenDD locations, not independent confirmation.
The protocol uses eight observed and 12 forecast native steps. The
recording-level 2% risk rule relative to constant velocity (CV) is
unchanged. Undefined easy support does not pass. DroneCrowd stays at
15 calibration and 30 sealed confirmation groups; no confirmation scores
are used here. No new metric/time claim, deployment, Stage5C, or SMC.

At each outer location and for each of three seeds, six inner heads
held out one donor location each. The donor's score and its easy cut
exclude both that donor and the outer location. A full donor fit is then
scored on the outer location. The bank uses 25%, 50% and 100% interpolation
from CV to four-point least-squares velocity (OLS4). The policy can edit
at most two agents and leaves at least one agent on CV per joint query.
The independent control uses the same query edit counts, not the same
recipients or amplitudes. The frozen tangent reference is not budget matched.

| Policy | Edits | Risk passes | Undefined | Risk pass and beat both references | Gain vs CV (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Utility only | 5,361 | 84/606 | 207 | 52/606 | -0.190 |
| Donor-selected safety | 5,214 | 85/606 | 208 | 54/606 | -0.192 |
| Safety plus easy-support hurdle | 5,214 | 85/606 | 208 | 54/606 | -0.192 |
| Same-count independent utility | 5,214 | 85/606 | 208 | 54/606 | -0.197 |
| Frozen tangent (unmatched) | 30,075 | 522/606 | 30 | 427/606 | +0.306 |

The hurdle improved on OLS4 by +6.757 pp, but its CV contrast had a 95%
location-bootstrap interval of [-0.656, +0.125] pp. Its direct difference
over the same-count control was +0.005 pp [-0.004, +0.019]. The seven
locations are resampled after averaging seeds, rather than treating 606
recording-seed evaluations as independent samples. Donor calibration
selected the smallest nonzero support threshold (0.05) in all 21 outer
fits; adding it produced no measured advantage over safety-only.

**Hindsight diagnostic.** With the identical bank and query cap, a mixed
integer search using future labels verifies 170/202 distinct recordings
can pass risk and beat both references. It finds 19 risk-infeasible
recordings, three bounded below the positive-gain criterion, and ten
without easy reference support. This is not a learned policy result.
The gap supports testing better recipient/action ranking and supervision
on same-count swaps, not interpreting another threshold grid as deployment
evidence.

**Provenance.** Training and exact capacity are `fresh_run`: 147 native
arm64 Torch heads, 58,800 updates, CPU with four threads, no workers.
Replay is `cached_verified`: checkpoint hashes bound to the frozen heads,
identical held scores, zero updates. Independent external confirmation is
`not_run` because the controller did not qualify. See [scores](analysis.json),
[registration](registration.json) and [capacity](exact_analytic_capacity.json).

```sh
.venv-pytorch/bin/python scripts/run_m3w_selected_bridge_hurdle.py
.venv-pytorch/bin/python scripts/run_m3w_selected_bridge_hurdle.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_selected_bridge_capacity.py
```
