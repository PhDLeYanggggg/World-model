# Fixed-candidate recipient ranking

Prior same-budget hindsight attributed most regret to *who* receives a
new forecast. Test this without simultaneously optimizing *which*
forecast they receive. From the same causal expert menu, nominate one
nonidentity action per agent by short in-history forecast errors, and
apply exactly one quarter of its displacement from OLS4. This
deterministic past-only nomination and amplitude are shared by all
arms, including hindsight. Do not transfer the prior unrestricted
oracle gain to this new menu or amplitude.

All learned arms use the same source-held folds, seeds, architecture,
query draws and 600 updates. Compare pointwise Huber regression of
conservative utility, exact expected top-two utility under sequential
weighted sampling without replacement, and the latter plus query
context. For the last arm, seven causal motion/neighbor/proposal
features are centered against other current agents. Other arms zero
that block. The differentiable objective sums the two inclusion
probabilities times per-agent gain (not a classification of the best
expert); inference deterministically takes the two highest scores.
Gain targets are query-scale normalized from the past and debit
unobserved steps by trajectory disagreement with OLS4.

Controls: leaving OLS4 alone, matching counts by minimum change,
in-history replay confidence, and deterministic seeded uniform
selection. The oracle sees future labels only in the scorer, with
identical recipients available and the same count. The unit of
training resampling and evaluation is the query; statistical
intervals resample the nine fitting sources, not windows. Use 3,000
paired draws after seed averaging. Primary contrasts are top-two
versus OLS4, minimum change, and pointwise regression; the context
contrast uses the exact same top-two objective. Compare forecasts
directly for each contrast, rather than subtracting lower bounds.
Retain all 8/12 windows and report available ADE/FDE, easy harm and
support, hard, positive-action frequency, and fixed-nomination regret.

[Plackett](https://doi.org/10.2307/2346567) and
[Oosterhuis](https://arxiv.org/abs/2105.00855) provide the ranking
background. Neither the ranking distribution nor a generic ranker is
a claimed contribution. This experiment asks whether aligning a
ranking objective with a small intervention budget extracts
cross-source trajectory gains under partial outcome support.

Fitting-source experiment only. Same original 2% rule and sealed
DroneCrowd confirmation. No new policy threshold, external role
access, physical/metric/time claims, deployment, Stage5C or SMC.
