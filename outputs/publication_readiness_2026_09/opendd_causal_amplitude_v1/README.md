# Causal Recipient and Amplitude Allocation

**Result: no useful joint advantage at equal realized budget.**
The sparse controller loses recording support. Distributing the same
movement over more targets restores many passes. Simple fixed half-scale
shrinkage is a stronger development control, not a qualified policy.

[Registered protocol](protocol.md) | [Fresh readout](analysis.json) |
[Distributed-budget control](amplitude_spread.json) |
[Half-radius shape comparison](half_radius_shapes.json) |
[Manuscript supplement](../evidence_manuscript_v33/README.md)

## Registered matched comparison

63 new tree heads (seven held OpenDD locations, three producer seeds and
three label/feature arms). Training predictions are from frozen producers
that exclude both the donor and held locations. All operating arms
allocate quarter, half or full incumbent departures from CV using past-only
inputs and a predicted ledger. Rolling footprint geometry uses the eight
previous queries and current histories, plus a count of prior queries.
Risk is recomputed from released outcomes after inference. Future-label
availability is not a runtime feature.

| Policy | Lower gain vs CV (pp) | Edits | Original passes /606 | Undefined selected easy |
| --- | ---: | ---: | ---: | ---: |
| Full incumbent | +0.429 | 30,075 | 455 | 30 |
| Direct debit regression | +0.305 | 14,280 | 404 | 87 |
| Factorized, without footprint prefix | +0.307 | 14,314 | 406 | 81 |
| Factorized with prefix, independent | +0.295 | 14,275 | 400 | 87 |
| Factorized with prefix, joint | +0.204 | 14,166 | 407 | 78 |
| Joint's budget, matched independent | +0.207 | 14,166 | 403 | 86 |
| Joint's budget, matched additive | +0.206 | 14,166 | 407 | 78 |

Each matched control uses the same per-query multiset of actual maximum
departures and the same edit count, not just the same blend coefficients.
Joint lower gain vs matched independent is -0.0044 pp
[-0.0096, +0.0001], and vs matched additive -0.0017
[-0.0057, +0.0004]. The forecast compression proxy is 10.17,
12.03 and 10.16, respectively. This is no established utility or
geometry advantage over additive control, and the proxy does not
measure physical safety.

The wholly unknown event is rare: 159 target-seed rows in the 12 views
containing positives. In those views mean AP increases from 0.0077 to
0.0089 with the rolling footprint (mean prevalence 0.0066), too small
to establish a useful support predictor. Factorization and footprint
geometry do not close the actual risk gap.

## Post-hoc: spread the same movement instead of pruning

For each query, distribute the sparse joint arm's total maximum
departure over all eligible paths at a common coefficient. This uses
no outcomes but depends on the joint query budget. It matches movement
mass, not edit count. The separate fixed-half rule uses no joint
controller and has a slightly higher total movement budget.

| Rule | Edits | Total radius | Passes /606 | Undefined selected easy |
| --- | ---: | ---: | ---: | ---: |
| Sparse joint | 14,166 | 282.773 | 407 | 78 |
| Spread query budget | 29,840 | 282.773 | 544 | 30 |
| Fixed half amplitude | 30,075 | 291.634 | 547 | 30 |

The spread recovers 137 net passes. Its paired lower utility vs sparse
joint is -0.0305 pp [-0.0587, -0.0053]; the two CV-relative point
estimates (0.205 and 0.204) cannot be subtracted as if missing-step
disagreement were the same for the two pairs. Fixed half amplitude
gains +0.219 pp [+0.145, +0.325] vs CV; 29 defined failures and 30
undefined views remain. No policy passes the full requirement.

At the same half radius and same recipients, the previous pointwise
shape beats fixed half by +0.031 pp [+0.014, +0.053] but drops to 542
passes. The non-canceling risk shape has 551 passes but loses -0.037 pp
[-0.053, -0.023]. A matched half-radius hindcast direction passes 552
but loses -0.263 pp against the half-radius neural incumbent.
None jointly improves utility and risk against that simple control.

**Decision:** keep fixed half as a new development comparator. A
useful next study must preserve selected easy support and compete with
simple shrinkage, not merely outperform sparse selection. The joint
geometry penalty has no demonstrated incremental contribution here.
Intervals use 3,000 paired location draws after averaging seeds.
No independent evidence, deployment, new units claim, Stage5C or SMC.

## Reproduce

~~~sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_causal_amplitude.py register
.venv-pytorch/bin/python scripts/run_m3w_opendd_causal_amplitude.py run
.venv-pytorch/bin/python scripts/run_m3w_opendd_causal_amplitude.py replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_amplitude_spread.py
.venv-pytorch/bin/python scripts/audit_m3w_opendd_half_radius_shapes.py
~~~

Native arm64 CPU, four Torch threads, no workers, about 21.5 seconds
for fresh fitting and scoring. Hash-verified no-fit replay is exact.
Twenty-eight targeted tests passed. No CREATE job or DroneCrowd
confirmation access.
