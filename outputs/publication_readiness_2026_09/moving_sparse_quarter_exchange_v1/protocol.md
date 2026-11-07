# Causal Sparse Exchange At Quarter Amplitude

Use the original quarter roster and forecasters on 34 exposed recordings,
14 sources and three forecast seeds. Keep each query's original
intervention count and its quarter amplitude. At most one different
agent replaces an original intervention in a query, at most four such
edits per recording. Preserve all remaining original trajectories.

Fit one 32-wide paired cost model per fold (440 updates, batch 256).
Use separate causal descriptors for insertion and deletion, with a
relative past-motion scale. Train on paired lower utility, all and easy
risk-excess changes, signed easy risk change and easy-support change.
The first four targets use the mean of the two past motion scales.
Sample up to 128 pairs per query without using labels to choose pairs.
Fitting inputs use source-excluded estimates of the original policy
costs. No saved oracle choices are used.

Compare two rankings from the same learned costs. Both penalize
predicted positive risk changes and reject negative predicted gain.
The support arm adds .25 times the predicted support change clipped
to [-1,1]. Apply the first four query decisions above the threshold
in time order, not the top four across the completed recording.
Report threshold-0 diagnostics. Calibrate both arms separately from
{0, .025, .10, .25} on six donor sources only, requiring new passes,
no lost passes, no added signed-easy violation and nonnegative paired
lower gain. Fall back to the original roster when none qualifies.

Fit on seven sources, calibrate on six, and score the excluded source.
Compare with the frozen quarter and dense gain/harm controls, requiring
exact replay. Primary contrast: calibrated support arm vs quarter.
Report source-group bootstrap and original 2%/support risk. These are
development results. Do not assign roles or open external outcomes.
