# Local OLS4 Proposal: With And Without Neighbors

Test whether learning a small local correction to the strong causal
OLS4 forecast produces more accurate candidate trajectories than OLS4
itself, and whether neighbor histories explain any improvement. This
is a proposal generator experiment, not a new controller.

Use the exposed Moving A development population, eight observed and
twelve future annotation steps at native image scale. Within each of
three seeds and each held source, fit on the same seven prior fitting
sources, select amplitude on the same six disjoint donor sources and
freeze outputs on the held source. A two-vector quadratic-in-time
displacement is bounded by half the historical mean step speed
(with a one-pixel scale floor), times the forecast step. Both models
have the same parameter count and row draws. The self model zeros
seven context inputs, including the nearest past neighbor's relative
position and velocity and the query mean past motion. Features contain
no future positions or label masks.

Fit 240 AdamW updates per cell (batch 128), sampling sources and queries
with equal total mass, with observed-step ADE change and a displacement
penalty for wholly unknown outcomes. Save every 80 steps. On the six
donors choose the amplitude in 0, 0.25, 0.5, 1 that maximizes mean
source completion-lower ADE gain relative to OLS4, normalized by the
same CV reference mass. Break ties toward the smaller amplitude. Keep
both the fixed amplitude-one and the donor-selected results.

Primary paired comparisons are selected neighbor vs. OLS4 and selected
neighbor vs. selected self. Bootstrap 3,000 sources after aggregating
seeds and score all 102 recording-seed views with the original CV risk
reference, including missing outcomes and the original 2% rule.
No new policy threshold or deployment; a useful candidate is not
automatically a safe policy. No calibration or confirmation data.
