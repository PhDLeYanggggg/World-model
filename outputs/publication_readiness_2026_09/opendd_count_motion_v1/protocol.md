# OpenDD development: same realized count and no more total movement

Keep each recording-seed's original online edit count, nested nominee prefixes,
chronological token rule, 2% risk checks and CV/OLS4 references. The additional
per-recording ceiling is the online sum of the per-edit maximum Euclidean
displacement from CV over the 12 forecast steps. All forecast paths and their
individual amplitudes are frozen. The solver may spend less movement, not more.
This is an additional control, not a new definition of physical safety.

Use two offline reallocations under these same constraints:

1. Label-blind: maximize the sum of frozen count-logit increments against
   zero edits. It sees all query scores and the final online spend, not labels.
   Score it afterward using the original evaluator. This is not an online arm.
2. Hindsight: maximize the smaller of the two conservative gains under the
   unchanged original risk checks, using exposed development outcomes. Verify
   the witness with the original evaluator. Missing support and unresolved
   optimizations are not successes. Zero edits are not a risk pass.

Primary readout: dual-pass counts, paired equal-location differences and 3,000
location bootstrap draws (seed 817, seeds averaged within location). Also report
the leave-one-location-out range. Keep every one of the original 606 views;
no new fitting, retuning, confirmation access or deployment.

Numerics: remove individually unaffordable options without removing their
eligible-query credits. Use a 1e-9 relative movement roundoff allowance,
disable MILP presolve, and recheck count, movement, tokens and the original
trajectory scores on every returned plan.
