# Streaming Risk Budgets Do Not Rescue Recording-Level Safety

**The causal selector rescued some previously failing recording views,
but lost more passes than it gained.** Joint geometry reduced a forecast
distance proxy against independent selection, at an accuracy cost and with
no benefit over the matched additive control. This is a development result,
not a change in deployment or confirmation status.

## Fixed Forecast, New Decision Rule

I kept the previous uniform calibrated forecast and fitted 21 new
Extra Trees risk heads (seven held locations, three forecaster seeds).
Each head sees predictions from six donors; each donor prediction comes
from a frozen forecaster that saw neither that donor nor the outer site.
The five predicted moments cover expected gain, all/easy positive harm
including unknown-outcome debit, and all/easy reference mass. All input
features come from observed motion and the fixed predicted paths.

I compared query-local selection, a streaming per-recording ledger, and
the same ledger with a joint pair-compression penalty. The ledger sees
only earlier predicted reservations, never observed future errors or the
remaining length of the recording. The unchanged original risk rule is
recomputed after rollout for all 606 flight-seed views. Forecast-side
probabilities or moments do not certify that rule.

| Policy | Lower gain vs CV, pp [95% site bootstrap] | Switches | Worst positive easy harm (site) | Flight-seed risk passes |
|---|---:|---:|---:|---:|
| Frozen uniform blend | +0.474 [+0.304, +0.735] | 30,075 | 2.46% | 450 / 606 |
| Query-local | +0.289 [+0.180, +0.451] | 14,035 | 0.70% | 425 / 606 |
| Streaming prefix | +0.325 [+0.213, +0.478] | 14,266 | 0.80% | 399 / 606 |
| Streaming prefix, joint geometry | +0.236 [+0.158, +0.345] | 14,193 | 1.00% | 401 / 606 |

The switch counts sum over three seeds and 10,475 targets. Intervals use
3,000 paired resamples of seven locations after averaging seeds, not
independent windows or cities. The joint forecast's observed ADE/FDE
is 0.6066/1.3500 versus CV's 0.6083/1.3534 in publisher units. CV has
no edit-supported risk ratios and is not counted as a safety pass.

## Matched Geometry and Support

At the same **14,193 edits, matched within every query**, joint selection
gains **-0.117 pp** [-0.176, -0.067] relative to independent utility and
**-0.015 pp** [-0.034, -0.0002] relative to additive single-edit geometry.
The proxy (summed mean reduction in near-pair forecast distance) is 11.83
for joint, 25.02 for independent, and 11.52 for additive. Thus there is an
accuracy/geometry tradeoff against independent selection, and no observed
geometry advantage over additive. The controls inherit the causal counts
but do not enforce the ledger; the comparison is not equal realized risk
and says nothing about physical collisions.

| Change relative to frozen uniform | Query-local | Prefix | Joint |
|---|---:|---:|---:|
| Previously failing, now passes | 46 | 36 | 36 |
| Previously passing, now fails | 71 | 87 | 85 |
| Of the lost passes, selected easy reference becomes undefined | 32 | 34 | 34 |

The audit pairs each recording and seed; rescued passes are a subset of
the 112 hindsight-feasible uniform failures, not new sites. The 30 views
with no full easy reference remain unestimable. Sparse switching can
lower site-wide harm while removing the very reference support required
by the per-recording rule. In the joint arm, lost easy support explains
34 of 85 regressions; it is not the only failure mechanism.

**Decision:** do not carry this prefix/joint policy into confirmation.
Keep the improved forecast and address selected easy reference support
explicitly in the next decision head, rather than increasing the geometry
penalty or declaring no-edit recordings safe. Do not alter the original
2% rule. No access to the 30 sealed DroneCrowd groups, Stage5C or SMC.

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_prefix_risk.py
.venv-pytorch/bin/python scripts/audit_m3w_opendd_prefix_transitions.py
```

The local arm64 run took 8.56 seconds with frozen trajectory checkpoints.
It saved one risk-head checkpoint per site and seed and logged its PID.
The separate readout matched byte for byte. No models or underlying data
are distributed.

[Protocol](protocol.md) | [Per-site results](analysis.json) |
[Paired recording transitions](flight_transitions.json)
