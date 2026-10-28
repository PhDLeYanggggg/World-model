# Constraint Gradient Diagnosis: Complete

Result source: fresh_run frozen-model autograd calculation on verified TRAIN
packets; parent heads cached_verified. All 216 fixed 128-query monitors replay.
No optimizer step, new development inference or independent labels were used.

For task-risk, median weighted easy-risk gradient/data-gradient norm is
**0.000698029**, about **0.0698%**. Total weighted risk-gradient ratio is 0.0715774.
Median regret/easy-gradient cosine is -0.461932: their local descent directions
can oppose each other. These are final-parameter descriptions, not training-path
causal effects. The data gradient combines primary regression and decision regret.

Median TRAIN budget scales are 0.02 for all and 0.000418546 for easy. Soft excess
remains positive in 38/72 all and 61/72 easy monitors; the original hard action
gives positive excess in 25/72 all and 52/72 easy. Mean soft action is 0.125066
versus hard action 0.325990. These TRAIN quantities are distinct from development
readout statistics, and do not describe every TRAIN query.

This supports testing whether constraint units impair optimization, not claiming
that rescaling will fix safety. The [matched repair](../european_scaled_risk_v1/protocol.md)
divides constraints by fixed positive TRAIN reference budgets in primal and dual
updates. The 2% definition and zero set remain unchanged, while optimization
conditioning and effective dual capacity change. All other settings stay fixed.
Both arms are freshly trained; the unscaled control must exactly reproduce all
72 prior task-risk fits. No model is promoted from this diagnostic.

The [summary](summary.json) and [hash inventory](complete.json) retain all arms.
This is not a new safety theorem or independent world-model evidence. Stage5C
and SMC remain disabled.
