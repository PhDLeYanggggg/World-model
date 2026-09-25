# Producer-Conditioned Gain/Harm Heads

Registered development experiment, 2026-09-25. Freeze and push this protocol with code/config/tests before any new fitting. All heads and both producer-branch decision banks must complete before new readout. No threshold search or deployment selection.

## Hypothesis And Prior Evidence

The earlier `european_producer_transport_v1` already found that substituting two-source for four-source forecasters under frozen heads is not a reliable repair. The latest support-factorization study found that history and disagreement support boxes both discard useful predictions. This study is not another producer substitution or percentile search. It tests whether a learned controller benefits from identifying which already-trained producer bundle made its candidate and floor forecasts.

## Fixed Data And Models

Twelve opened EuropeanSquares source localities, 8 observed / 12 predicted steps, raw annotation stride12, image pixels and released detector tracks. Preserve three source folds, three seeds 17/29/43, train-only normalizers, all/easy events, baseline choices and the 2% predicted-risk budget. Independent model selection, calibration and confirmation roles stay closed.

Within each four-source fitting fold, two fixed disjoint halves contain two sources each. Fitting rows are scored by the opposite half's neural forecaster and protected CV/damping floor. All eight excluded development localities are evaluated separately with each same two-source producer bundle. No averaging or picking the favorable half. The earlier two-source neural forecast banks are hash-verified and reused; two-source floor readout and checkpoint prefixes are freshly computed. The original four-source floor and stop policy remain common outcome anchors, so changing the fallback cannot silently improve the denominator.

## Matched Training

Three new arms append two columns to the same 380 past/rollout features, making 382 inputs in every arm:

1. Global: two zero columns.
2. Producer: one-hot actual bundle half, metadata known when predictions are produced.
3. Placebo: one-hot deterministic outcome-independent row-key hash, only a capacity/noise control.

Each group/arm trains bounded utility and occurrence/severity ranking-risk heads. Exactly 18 fold/seed/event groups, 108 heads and 216,000 updates. Width64, batch256, 2,000 final updates, identical optimizer/sampler/labels/rollouts and fixed fitting-derived ranking denominator. No best-checkpoint selection from readout. Checkpoint/heartbeat every200 updates. A first-group 100-step utility/risk pilot resumes within the same fixed budget; no pilot outcomes are read.

This uses the registered fitting-fixed normalizer as a prespecified scope, not a new search over normalization. The batch-normalized branch is not retrained. Forecasters are not retrained. Thus the new neural learning is controller learning, not trajectory-dynamics learning.

## Controls And Evaluation

Per group and producer branch, evaluate global, producer, placebo, wrong-tag producer diagnostic, and the earlier unmodified 380-feature floor-target head on the identical branch forecasts. This is 180 views. Also retain the earlier four-source stopping-protected policy and raw two/four-source predictors as anchors. Wrong-tag inference flips the producer flag only after fitting and before readout; it is never a deployment candidate.

Primary comparisons: producer vs new dimension-matched global and placebo, with identical candidate and fallback forecasts. Report all/easy/hard/complete ADE and endpoint FDE, intervention rates, unknown-label support, zero-CV harm, worst-locality/tails, gain/harm score error and realized selected harm. Use 3,000 paired source-locality bootstrap resamples; all 36 producer/group comparisons are dependent development views, not independent confirmations or multiplicity-adjusted wins.

Freeze all 180 decisions before reading new outcomes. Recompute choices independently, replay both producer prefixes from each trained checkpoint, verify identical supervised draw counts and label hashes across arms, and verify zero unknown supervised samples. Report a full matrix, not only a favorable half/seed.

## Attribution Limits And Decision Rule

Producer identities are confounded with the two fitting-source cohorts; a tag effect can encode cohort-specific response, not a uniquely identified causal transport mechanism. Matched global/placebo controls and wrong-tag sensitivity constrain interpretation but do not remove this limitation. Replacing four-source with two-source forecasts changes quality as well as producer distribution. Only identical-forecast comparisons isolate the new head change.

No promotion without robust paired gain, <=2% observed positive-easy degradation and no observed zero-CV harm. Even those development criteria do not open confirmation or certify physical safety. Negative or seed-dependent results must be retained. Image-pixel raw-frame only; not historical t50, seconds, metric, human gold, true3D or foundation. Stage5C and SMC remain off.
