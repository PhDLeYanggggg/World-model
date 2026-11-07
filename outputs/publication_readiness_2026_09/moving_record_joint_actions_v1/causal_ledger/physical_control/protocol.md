# Reoptimized Physical-Only Control

The main ledger result identifies a decoding effect, not a contribution of
learned trajectory candidates. Fix the input costs, time order, 2% risks,
query cap, solver and A population. Use the physical-only pointwise policy
to set the count in each query. This guarantees a common causal feasible
witness. Reoptimize the physical ledger at those counts using all five
causal alternatives to CV. Separately reoptimize the full ledger with the
same counts, adding EqMotion, PCT and the self-history kinematic candidate.
Do not simply delete learned actions from a fixed full solution.

Primary direct paired completion-lower comparison is full versus physical
ledger. No additional fitting or tuning. Freeze all three seeds before
outcome scoring, record action-family counts and every original risk view,
and bootstrap 3,000 times at source level after seed averaging. This is an
exposed-A matched control, not an independent or deployment result.
