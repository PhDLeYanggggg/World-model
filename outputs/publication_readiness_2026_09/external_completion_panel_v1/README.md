# Two ways an external comparison can remain uncertain

I use only the frozen WEPDTOF and Songdo scalar receipts. Both
comparisons hold the edit budget fixed and compare joint with
independent selection. A positive number favors the joint rule.
Gains use the original observed CV error denominator for each
study; they are not pooled.

| External contrast | Missing-label bounds for this sample (pp) | 95% source bootstrap for the lower endpoint (pp) |
| --- | ---: | ---: |
| WEPDTOF, 10 of 13 estimable sources | [-0.045010, +0.017596] | [-0.113230, +0.012138] |
| Songdo, 14 source components | [-0.000442, -0.000102] | [-0.001499, +0.000237] |

The WEPDTOF sign is not identified in the observed source sample.
In Songdo the mean is negative even at the optimistic endpoint for
the 14 observed components. However, the source bootstrap of that
endpoint is [-0.000545, +0.000266] pp. Dropping one component at a
time gives upper points from -0.000209 to +0.000084 pp. The data do
not establish a negative population effect either. Conditional
missing-label bounds are not sampling confidence intervals.

The Songdo calculation recovers the optimistic endpoint as twice
the observed contribution minus the conservative endpoint, within
each matched component. It then applies the same 3,000 draws to
all three columns. Replaying the observed and conservative columns
reproduces the published points and intervals. The
[readout](readout.json) also preserves the additive and same-family
controls. The optimistic Songdo endpoint is -0.000108 pp against
additive, but positive against the more restricted same-family control.
Those contrasts do not change any release decision.

WEPDTOF remains 0/45 joint dual passes. Songdo remains 0/480 joint
releases; the independent rule releases 15, four of which pass risk
and both accuracy references. The original 2% checks were unchanged.
The audit opened no annotations, refit no model, and did not reuse
pNEUMA for selection or evaluation.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_external_completion_panel.py
```

[v190 paper and figure](../evidence_manuscript_v190_external_bounds/README.md) |
[v20 anonymous replay](../external_completion_review_supplement_v1/README.md)
