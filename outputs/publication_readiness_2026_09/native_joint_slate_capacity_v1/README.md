# Capacity Of Frozen Whole-Query Forecasts

**Finding:** The current action slate constrains both recording risk and
accuracy against OLS4. A better selector cannot fix every view on this
same slate, even if it could see the labels.

The [registered diagnostic](protocol.md) uses 34 exposed development
recordings, 14 source groups and three actual forecast seeds. The
102 recording-seed views are not independent recordings. For each
view, a solver may abstain or use one intact proposal per query.
It maximizes conservative utility under the original all/easy
positive-harm caps, signed easy net cap and positive easy reference.
Two objectives are evaluated separately, using CV or OLS4 as the
comparator and the same CV denominator. Choices are never exported.

| Hindsight slate | Risk-feasible views | Infeasible views | Positive lower gain vs CV | Positive lower gain vs OLS4 | Feasible but nonpositive OLS4 upper bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint proposal only | 60 | 42 | 60 | 25 | 35 |
| Any of the four frozen proposals | 61 | 41 | 61 | 27 | 34 |

No view is unresolved. The solver proves its optimum in all
feasible views; those witnesses are rechecked with the original
scorer. Every recording has easy reference in the full label
roster, but 19 views have no easy reference in **any** four-option
proposal. Of the 41 infeasible views, 22 are ruled out by the joint
constraints even though at least one proposal supplies easy
reference. Increasing the number of choices among these four
options adds just one risk-feasible view and two positive OLS4 views.
The 34 nonpositive upper bounds concern the original *conservative*
gain under missing labels; they are not a complete-label error claim.

This differs from [best-of-K joint trajectory evaluation](https://arxiv.org/abs/2305.06292):
here the oracle diagnoses a fixed policy action space under a recording
constraint, rather than measuring a predictive distribution or choosing
a deployable output. It identifies where forecast geometry and the
available proposals need work, without explaining every learned
decision error or proving a limit on other model classes.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/audit_m3w_joint_slate_capacity.py development
.venv-pytorch/bin/python scripts/plot_m3w_joint_slate_capacity.py
```

[Development result](development.json), [execution receipt](development_execution.json)
and [figure receipt](figure_verification.json). The vector figure stays
local; the plotting source and light counts are committed. The optional
`calibration` phase is separate and requires the original score and
replay. It cannot choose a policy or open confirmation recordings.
No outcome access to the confirmation role, deployment, Stage5C or SMC.
