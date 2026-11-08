# Exact Three-Reference Capacity and Missing-Step Attribution

The [guarded paired selector](../tumdot_incumbent_selector_v1/README.md)
increased local risk-and-utility passes but lost conservative utility
against the fixed hindcast. I replay the frozen predictions exactly,
without changing the nine previously exposed recordings, three
seeds, recipients, or per-row peak displacement. This is one
campaign, not independent confirmation.

All entries below use the **same full CV reference mass** within a
view, so the three contributions sum to the full paired contrast.
Available gains and missing-step disagreement are not pooled as
observed outcomes. There are 1,259 complete, 180 partial, and 17
wholly missing rows before repeating each recording with three seeds.

| Half-radius guard versus hindcast | Rows across seeds | Changed paths | Observed pp | Disagreement debit pp | Lower pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Complete | 3,777 | 1,315 | +0.01782 | 0 | +0.01782 |
| Partial | 540 | 113 | -0.00243 | 0.06859 | -0.07102 |
| Absent | 51 | 8 | 0 | 0.00390 | -0.00390 |

Partial rows supply almost all the debit, but the complete-row
contribution is already negative at P3 (-0.04364 pp). Complete-only
scoring would conceal that the full-roster comparison loses.
Masks and future positions are not available to the frozen chooser.

## What the Menu Can Achieve

For each active row select exactly one of six paths, keeping all
inactive rows at CV. Each action adds its exact paired missing-step
lower gain against CV, OLS4, and hindcast. Maximize the smallest
of the three normalized gains subject to the same selected all/easy
positive-harm budgets and full-easy signed degradation bound.
Reference support is unchanged by direction at a fixed edited
roster. Every solution is checked by the original scorer at the
unchanged 2% gate and the original radius. No held labels are used
by any deployed or causal candidate.

| Radius | Verified strict passes over all three references | Optimal solves | Mean minimum lower gain (pp) |
| --- | ---: | ---: | ---: |
| Half | 27/27 | 27/27 | +0.37572 |
| Quarter | 24/27 | 27/27 | +0.15010 |

The strict positive check uses a fixed numerical margin of
0.000001 pp. The oracle results describe this menu and recorded
outcomes only. They do not certify learning or an independent risk
guarantee. A [donor-only follow-up](../tumdot_joint_teacher_v1/README.md)
tests whether these better assignments are learnable.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_tumdot_three_reference.py
.venv-pytorch/bin/python scripts/audit_m3w_tumdot_three_reference.py --replay
```

`fresh_run`: 54 exact solvers and score attribution, 1.712 s,
PID 91637, CPU. Per-cell private receipts and heartbeat.
`cached_verified`: frozen choices, exact repeats. Private outcome-aware
assignment arrays are not exported. No CREATE job, deployment,
independent confirmation, Stage5C or SMC.
