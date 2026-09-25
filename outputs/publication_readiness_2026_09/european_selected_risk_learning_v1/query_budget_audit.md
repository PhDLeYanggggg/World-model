# Frozen Query-Budget Accounting

Fresh_run posthoc accounting of frozen decisions, not a new policy or selected
result. No action, threshold, role or model changes. Cached label arrays and
row IDs are hash-checked. Future label support is used only in this diagnostic.

## Conditional Harm Is the Main Measured Defect
Full selected-joint, across 72 dependent locality views:
- Median predicted/actual selected easy-harm numerator: **0.56124**.
- Median predicted/actual supported easy reference mass: **0.96132**.
- Unknown-future share of predicted easy mass: **0.7441%-4.8428%**.
- 2,602 of 155,908 active query views spend more predicted supported-row harm
  than 2% of their supported-row predicted reference mass. They still respect
  the registered all-indexed-row budget. This is a hindsight support mismatch,
  not permission to remove rows using future availability.

For the worst observed full selected-joint easy-risk case,
fold1/seed29/controller2, locality074:

| Easy-event quantity | Prediction | Observed on supported labels |
|---|---:|---:|
| Reference mass, supported rows | 37,245.63 | 32,694.64 |
| Selected positive harm, supported rows | 184.75 | 1,862.20 |
| Budget harm ratio, all indexed predictions vs supported outcome | 0.4986% | 5.6957% |

Unknown-future agents supply only 1.4143% of predicted easy mass in this case.
Supported easy reference mass is overpredicted by about 13.9%, whereas selected
easy harm is underpredicted by about tenfold. Unknown support alone is therefore
not an adequate numerical explanation of this locality's violation. This
diagnostic localizes the error; it does not identify whether input information,
optimization, target parameterization or domain shift caused it.

## Controls
Full mean-joint has median selected easy-harm prediction/actual **0.47650** and
supported reference prediction/actual **0.99323**. Motion-only selected-joint
has **1.07064** and **0.94492**, respectively, but still violates the easy
constraint in three dependent locality views. Medians do not certify tails,
and action sets differ between arms.

All frozen joint decisions satisfy their predicted query budgets. The observed
violation is thus not repaired by merely replaying or reordering the same sums.
The next controlled repair should focus on conditional easy-harm occurrence and
severity, not assume global denominator inflation is the whole problem.
No inference feature may contain future support, future easy labels or costs.

See query_budget_audit.json for all 72 pair/arm groups. Image-pixel annotation
steps, detector-derived development labels only. No metric/seconds, human-gold,
physical-safety, true3D or foundation claim. Stage5C/SMC remain off.
