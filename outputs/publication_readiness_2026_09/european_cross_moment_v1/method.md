# Conditional Risk Target and Normalization

## Why the Target Matters

Let B be the nonnegative reference cost and H the nonnegative error increase
caused by an intervention. Deployment ranks a conditional ratio E[H|x]/E[B|x],
not E[H/(B+H)|x]. These are different quantities. The previous constructed
counterexample shows that realized-share ranking can prefer the opposite order;
it does not establish the cause of every empirical failure.

For a pair i,j, this experiment uses D=H_i B_j-H_j B_i. Under independent
conditional draws, its conditional expectation has the sign of the difference
in conditional ratios when both expected denominators are positive. The neural
score remains log(Hhat+epsilon)-log(Bhat+epsilon). Weighted logistic pair loss
uses abs(D) and sign(D). Labels and cross-products are detached; future outcomes
are fitting supervision only, never inference features.

The pair sign generally agrees with realized-share order. The intended change
is its magnitude weighting, not extra feature information. Exact ties can differ
because cross-products are computed in float64 rather than float32 ratios.

## Two Controlled Steps

1. Batch: replace realized-share weights with abs(D), retaining normalization by
   the current batch's weight sum. Control: the frozen supported-pair head.
2. Fitting: keep abs(D), replace the random denominator with a fixed mean sum
   estimated from the first 40 original fitting batches per head. Control: the
   newly fitted batch-mode head. Both start at the same original initialization.

Keep the other losses, forecast producers, 355 causal inputs, minibatches,
capacity, optimizer, 2,000 updates and 2% risk rule unchanged. Both complete
decision banks are frozen before either new outcome evaluation. All registered
variants are reported; this is not a validation search over two alternatives.

## What Is Not Proved

Neither mode has a consistency or safety guarantee here. Pairing is conditioned
on positive observed event mass; agent/windows can overlap and be correlated.
The fixed normalizer is estimated from finite training data, and the nonlinear
model minimizes several objectives. High-cost pairs can dominate gradients.
Weight concentration and effective pair count therefore accompany accuracy.
No outcome-dependent clipping, threshold change or coefficient sweep is used.

Full-policy changes combine ordering and intervention-count effects. Both
matched-count anchors and their common-support decomposition are retained to
separate these effects arithmetically. That decomposition is not unique causal
mediation. Forced-count arms can violate predicted risk limits and are not
deployment candidates.

All twelve European Squares localities are opened development. Four fitting
and eight complete-chain-excluded localities per fit do not restore independent
confirmation. Three seeds and 3,000 locality bootstrap resamples give conditional,
dependent, unadjusted uncertainty. Image-pixel obs8/pred12 rawstride12 is not
metric/seconds, human gold, physical safety, true 3D or foundation evidence.
Reserved roles stay closed; deployment, Stage5C and SMC remain unchanged/off.
