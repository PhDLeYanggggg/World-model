# Different Forecast Shapes at the Same Realized Radius

This is an OpenDD development comparison on seven locations, 202
distinct recordings and three seed views of the frozen intervention
policy. Every arm replays the same 1,457 recipient-time interventions.
Every candidate has the original per-row peak distance from CV; even
the individual CV versus OLS4 interpolation amplitudes are inherited
before constructing the bank. Unselected trajectories remain exactly
CV. Horizons are eight observed and twelve forecast native steps;
coordinates and original CV-relative 2% risk are unchanged.

The timing bank retains linear, early and late corrections along the
same CV-to-OLS4 direction, with the same final correction. The expanded
bank also contains a damped-speed trajectory, a longer velocity fit, and
a tapered turn estimated from observed velocity headings. The last
three are projected to the same per-row peak radius. Turn-rate motion
models are established [kinematic comparators](https://pmc.ncbi.nlm.nih.gov/articles/PMC9370855/);
the comparison here isolates path shape from who was selected, when,
and how far the prediction was permitted to move. It does not claim
novelty for the kinematics or verify physical collision safety.

The causal choice minimizes average forecast error on suffixes of the
eight observed positions, using prefixes of length four, five and six.
The consensus variant changes from the linear path only if all three
prefix checks prefer its alternative. No labels after the observation
window enter these choices. No additional fitting or tuning is involved.

| Policy | Original risk passes | Risk pass and beat both | Direct gain over anchor (pp) |
| --- | ---: | ---: | ---: |
| Frozen linear anchor | 87/606 | 61/606 | reference |
| Timing-only prefix | 121/606 | 68/606 | -0.011 |
| Expanded prefix | 112/606 | 77/606 | +0.026 |
| Expanded prefix with unanimity | 95/606 | 65/606 | -0.001 |
| CV-gain hindsight (not causal) | 217/606 | 190/606 | +0.159 |

The expanded prefix choice beats timing-only by **+0.035 pp**
**[+0.012, +0.065]** in the paired lower full-grid gain, but it does
not establish an advantage over the frozen anchor (+0.026 pp
[-0.027, +0.082]). Intervals use 3,000 paired draws over locations
after averaging seeds. The 606 recording-seed views are not 606
independent records. Among 1,457 choices, the expanded rule uses the
tapered turn 534 times, damped speed 275 and long fit 277 times.

**Exact shape-only ceiling.** With true future labels used only in the
solver, the timing menu admits 80/606 original-risk and dual-gain
witnesses; the expanded menu admits 190/606. Each witness is rescored
and preserves every active row and radius. Of 110 added witnesses,
80 were risk-infeasible for the timing menu and 30 had an upper bound
without strict positive gain. There are no unresolved views. Shape
cannot change the 334 undefined easy-risk views (172 no edits, 160
without selected easy references, two without any easy reference).
This establishes geometric capacity, not causal identifiability.

`fresh_run`: causal prefix controls and exact capacity. `cached_verified`:
frozen admission checkpoint, nominations and exact anchor replay.
No new training or deployment here; Stage5C and SMC remain off and
DroneCrowd confirmation remains sealed. [Metrics](analysis.json).

```sh
.venv-pytorch/bin/python scripts/run_m3w_radius_matched_turn_menu.py
```
