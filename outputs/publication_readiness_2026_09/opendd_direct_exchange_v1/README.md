# Exact-Action Learning Does Not Beat Fixed Half

**Outcome:** 21 new direct pair models do not establish an improvement
over fixed half or the earlier marginal guard. This is development
evidence, not a replacement or an independent result.

[Protocol](protocol.md) | [Paired results](analysis.json) |
[Choice and recording attribution](choice_attribution.json) |
[Marginal experiment and oracle](../opendd_support_exchange_v1/README.md) |
[Manuscript supplement](../evidence_manuscript_v34/README.md)

Use the same seven held OpenDD sites, three seeds and pair-excluded
producer forecasts. Every policy edits the same 30,075 target-seed
rows at the same query-total realized radius. The exact action
head scores a particular pair and transfer, not interpolated
single-target scores. Both approaches fit observed cost labels and
then select an action; neither uses the decision-loss training in
[Smart Predict, then Optimize](https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922).

| Policy | Conservative gain vs fixed half, pp [95% CI] | Exchanges | Original passes /606 |
| --- | ---: | ---: | ---: |
| Fixed half | Reference | 0 | 547 |
| Marginal guard | +0.0007 [-0.0011, +0.0032] | 1,904 | 547 |
| Direct utility | +0.0023 [-0.0027, +0.0085] | 5,459 | 548 |
| Direct guard (primary) | -0.0019 [-0.0054, +0.0015] | 2,493 | 547 |
| Direct event guard | +0.00003 [-0.000004, +0.000094] | 3 | 547 |

Intervals use 3,000 paired draws of held locations after averaging
seeds. The direct guard's paired difference vs the marginal guard
is -0.0041 pp [-0.0078, +0.0005]. The direct utility arm restores
one pass without losing one, but its utility interval crosses zero.
All preserve the 30 undefined views; both direct guards preserve
every individual original pass. The primary criterion fails.

## What limits the learned policy

Of 9,717 eligible queries across seeds, 8,550 admit a beneficial
nonincreasing-risk exchange under the **outcome-aware** exact
envelope. The direct guard selects 2,493 exchanges, of which 1,225
have positive lower gain and 1,098 have both positive lower gain
and no increase in any of the three original risk numerators.
This is 44.0% precision among selected exchanges and 12.8% capture
of the queries with an available good action. The .75 event gate
selects only three exchanges, one of which meets both conditions.
Higher rejection is not a substitute for better choice.

The fixed half control remains the development reference. Neither
learning the true pair action nor predicting its safety event
justifies more threshold search on this cohort. A further positive
claim needs a stronger past signal and independent support, rather
than assuming the hindsight envelope is learnable from these
features. The 2% definition is unchanged. No external scores,
deployment, DroneCrowd confirmation, Stage5C, or SMC.

## Reproduce

~~~sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_direct_exchange.py register
.venv-pytorch/bin/python scripts/run_m3w_opendd_direct_exchange.py run
.venv-pytorch/bin/python scripts/run_m3w_opendd_direct_exchange.py replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_direct_exchange.py
~~~

Fresh direct fitting and readout take 24.3 seconds on native arm64
CPU with four Torch threads and zero workers. Heads checkpoint
individually; both experiments replay exactly with no new fit.
Thirty-three targeted tests passed. No CREATE job was needed.
