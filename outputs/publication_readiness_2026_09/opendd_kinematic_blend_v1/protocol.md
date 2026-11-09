# Stronger simple causal controls on the vehicle roster

Keep the seven OpenDD vehicle development locations, existing
eight-observed/twelve-requested grid, label masks and CV/OLS4
references. Compare the completion-trained vehicle incumbent
against four simple paths: anchored median velocity, gradual
velocity smoothing, decaying acceleration and decaying turn.
All start at the observed endpoint and are limited to the same
two-observed-scale-unit peak deviation from CV as the incumbent.
The OLS4 reference remains unchanged.

Also fit a convex mixture of these bounded corrections with
nonnegative weights summing to at most one. The residual weight
stays on CV, which is feasible by construction. Fit on the same
six donor locations, 51,200 sampled histories and deterministic
draw order per seed as the archived vehicle model. Optimize
the completion objective: observed normalized displacement error
plus disagreement with CV on missing steps, using a fixed smooth
norm of epsilon 0.0001 for the convex solver. Only the four
weights are fitted, with no held hyperparameter choice.

Freeze all 21 site-seed solutions before held scoring. Recompute
the archived incumbent to ensure masks and formulas agree. Report
native ADE/FDE, p95, easy error and the original recording risk
counts; paired conservative gain versus CV and directly versus
the incumbent; and 3,000 paired bootstrap resamples of location
means after seed averaging. Fixed primitives do not have a
training seed; repeating them in the paired views is bookkeeping,
not replication. No changes to deployment, external labels, metric
units, the risk rule or the prospective Songdo split.
