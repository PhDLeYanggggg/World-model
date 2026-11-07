# Dense Quarter-Amplitude Intervention Allocation

Fix the previous 8/12 development corpus, forecast banks and constant
velocity reference. Use 34 already exposed recordings, 14 source groups,
and three forecast seeds. No external labels or independent confirmation.
Preserve the original reference support and 2% risk rules.

At one-quarter amplitude, change which agent and forecast family are
selected while exactly preserving each query's original intervention
count. All nonidentity candidate paths for an active query are eligible.
Leave queries with zero original interventions untouched. Learn dense
targets instead of the earlier rare recording-rescue indicator:
completion-lower gain, positive all/easy harm including missing radius,
and their selected-reference masses. Fit a shared 32-wide network with
Smooth L1 loss (440 updates, batch 256); normalize with fitting-only
moments and past motion scale. Features use the causal candidate bank,
candidate family, original action and source-excluded prior costs.

The primary contrast uses identical learned costs and action counts:
maximize predicted gain alone versus gain minus estimated all and easy
risk excess (price 1). Per-agent family maximization followed by top-k
agent selection solves this additive objective exactly; it is not a
pair-interaction model. Compare both to the frozen quarter reference.
Separately, select a price in {0, .25, 1, 4} on donor sources only: add
one or more passes, lose none, add no signed-easy violation, and retain
nonnegative summed paired lower utility. If no price qualifies, retain
the original quarter roster. A hindsight additive rank uses observed
targets only as a labeled diagnostic, not as a policy or upper bound
on safe utility.

Use the same seven fitting, six donor, one excluded source roles and
three forecasting seeds as the previous study. Aggregate with the
existing source bootstrap and retain the original exact quarter replay.
Report count changes, reference failures, all/signed-easy violations and
paired lower utility without promoting any model from exposed outcomes.
No deployment change, Stage5C or SMC.
