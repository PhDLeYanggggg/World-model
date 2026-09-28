# Decision After Fitting-Only Diagnostics

Result source: fresh_run. All 108 groups and 1,728 gradient batches replay
exactly on CREATE. This is an optimization diagnostic, not new model training
or a new held evaluation. The parent negative policy result is unchanged.

## What Changed Our Next Action

At initialization, the auxiliary-to-risk shared-gradient norm ratio has median
0.594 and none of 432 total gradients opposes direct risk. At the final
supervised states, that ratio is 213.257; auxiliary conflict occurs in 66/432
batches and total conflict in 60/432. Across all parameters, total conflict is
92/432. The imbalance is therefore measured, not inferred from loss magnitude.
It is also not a universal conflict: most supervised-final batches have a
positive risk projection. A small risk gradient can inflate norm ratios.

At marginal-only final states, applying the unused supervised objective would
oppose shared risk in 299/432 batches. This is a counterfactual diagnostic,
not the update actually used to train those marginal models.

Easy support is present, but heterogeneous: the median query-balanced easy
prevalence over repeated fitting-source views is 0.289; median realized easy
harm/reference is 0.1985, well above the unchanged 0.02 budget. Realized
within-budget easy examples have median fraction 0.6769. This neither proves
they can be identified from past inputs nor establishes an irreducible floor.

## Controlled Repair

The next experiment will compare unchanged equal-weight supervision with a
single risk-priority auxiliary norm cap, holding initialization, architecture,
sample chain, normalization, forecasts and training budget fixed. For all
parameters together, use g = g_risk + alpha*g_aux, with
alpha = min(1, 0.5*norm(g_risk)/norm(g_aux)); alpha is detached. Zero risk
gradient disables the auxiliary update. The constant 0.5 is fixed before any
new fitting or held readout, not searched on old held outcomes.

This retains at least half of the direct-risk gradient projection in an
infinitesimal Euclidean step by the Cauchy-Schwarz bound. It is not an AdamW
descent guarantee, finite-step guarantee, generalization guarantee, calibration
certificate or new optimization-method claim. Earlier four-cost auxiliary
projection experiments failed; their failure is retained. This new comparison
tests norm dominance in a different, easy-risk-factorized objective, not a
generic claim that gradient surgery works.

Training must retain all 108 groups and three existing seeds. Causal actions
must be frozen before the next development readout. The primary repair
contrast remains same-query-count allocation, with risk/easy/zero-CV screens
reported even when accuracy improves. All-risk and utility stay frozen. A
fitting-loss improvement alone cannot promote the policy.

## Boundaries

The original selected-risk primary remains incomplete. The twelve localities
are already-opened development sources with 4/4/2/2 producer/controller/fitting/
held-development roles; repeated views are not independent sources. Independent
selection, calibration and confirmation remain closed. Obs8/pred12, raw-frame
stride12, image-local detector silver only. No metric, seconds, human-gold,
physical-safety, true3D, foundation, deployment or submission-ready claim.
Stage5C execution and SMC remain disabled.
