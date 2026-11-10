# OpenDD: how much utility can direction alone recover?

On the frozen 780 edited recipients, fix the CV-relative norm at
each of the twelve prediction steps. Use the complete development
outcomes only in an oracle calculation. Compare the exact optimum
over constant angles from -90 to 90 degrees in 15-degree increments,
the exact optimum over the existing five analytic directions, and
an analytic per-step upper bound for smooth learned directions.

For the upper bound, at each observed step rotate toward the target
as far as the angle limit allows, independently of other steps.
At a missing step keep the incumbent, giving zero paired debit.
This attains the maximum direct conservative utility among paths
with the fixed stepwise norm and independent bounded angles.
The smooth learned head is a subset. Report also the unbounded
per-step direction optimum as a wider upper bound.

Report dual-pass counts for the utility-optimal paths only; these
counts are not upper bounds on what a risk-constrained optimizer
could achieve. Preserve the original 2% checks and 3,000 paired
location draws after averaging seeds. No training, external access,
or policy promotion.
