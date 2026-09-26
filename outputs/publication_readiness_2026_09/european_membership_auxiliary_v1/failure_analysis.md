# Auxiliary Cost Failure Analysis

## Material Passport

Fresh registered source-development training/readout, not independent test.
288 heads and576,000 updates complete. Strong-comparator and tail gates fail.
All negative source assignments and motion-only findings are retained.

## Supported Findings

1. Implementation/budget mismatch is not the observed explanation: cost-only
   held predictions equal the original exactly; normalization, RMS scales,
   draw streams and budgets match. Model size is24,901 parameters in both arms.
2. The auxiliary branch learns its task: full-view median held membership
   AUROC0.86123. Therefore a flat/untrained auxiliary branch does not explain
   the cost failure. Membership probability remains absent from cost decoding.
3. Cost improvement is heterogeneous: one positive, one negative and four
   overlapping full easy-harm intervals vs original; point range -8.13% to
   +1.04%. Forty of72 dependent views improve, which is not the same as a
   stable source-assignment improvement or a count of independent wins.
4. The main cost fit improves in only21/72 full views;12 of these do not
   transfer. Both fitting tradeoffs and transport instability are present.
5. Tail selection remains unreliable: one full top10-harm-capture interval
   is negative. Better Brier/AUROC for membership cannot substitute for cost
   magnitude, tail capture or independently calibrated policy risk.
6. Exact reference denominator preservation prevents a denominator change
   from manufacturing the observed comparison. Full/motion populations
   differ, so their contrast alone proves no modality contribution.

## Hypotheses, Not Established Root Causes

Shared-encoder gradient conflict, unequal task scales, label noise, missing
severity-predictive context and finite training support remain plausible.
This experiment does not identify which is causal. In particular, a useful
classification representation need not retain what predicts rare large harm.
The earlier factor attribution also showed that unrealizable true membership
substitution alone would not consistently repair conditional severity.

## Next Discriminating Check

Use frozen membership_aux checkpoints and their existing fixed fitting
batches. Compute separate cost/BCE gradients on shared encoder parameters,
their cosine and norm ratio, retaining all source roles/seeds/pairs. No new
held outcome, threshold, model fit or coefficient sweep. If gradients conflict,
register one matched decoupled-training control; if not, investigate severity
target support and representation sensitivity before changing capacity.
These are proposed next actions, not results from this round.

No new trajectory gain, deployment or world-dynamics contribution is claimed.
Historical Stage35/37 leakage-affected scores remain exploratory. No metric,
seconds, human-gold, true3D or foundation claims. Stage5C/SMC off.
