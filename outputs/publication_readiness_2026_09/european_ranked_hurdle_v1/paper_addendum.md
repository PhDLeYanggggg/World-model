# Direct Ordering Supervision Does Not Establish Safe Neural Superiority

## Motivation and Method

The preceding matched-coverage analysis showed that occurrence-severity
supervision changes intervention count without consistently improving risk
ordering. We therefore add a controlled pairwise ranking auxiliary to the same
three-output risk model. Forecasts, utility scores, features, initialization,
optimizer, training-only scalers and balanced-locality minibatches are fixed.
Both neural and causal-damping candidates receive the same treatment.

For fitting-only reference-error and positive-harm labels B,H, define q=H/(B+H)
on supported nonzero-mass rows. Within each minibatch/locality, cyclic pairs
receive logistic ordering loss weighted by abs(q_i-q_j), normalized by total
pair weight. Undefined pairs and ties have no ranking term, but retain their
original moment, occurrence and severity supervision. The model ordering score
is log(Hhat+epsilon)-log(Bhat+epsilon), with epsilon=1e-6 in training cost units.
The auxiliary coefficient is1, fixed before training with no sweep.

Pairwise logistic ranking is established by
[Burges et al., ICML2005](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf).
The weighting and target above are our experimental adaptation, not a claim that
RankNet supplies a calibrated risk controller or that this loss is novel.

## Evaluation

Three source-role rotations,three seeds,two candidate predictors and two risk
events give36 new heads and72,000 optimizer updates. Each head has22,979
parameters. Each fit uses four fitting and eight complete-chain-excluded
localities, all drawn from twelve previously opened development localities.
Every216 full/common/matched-count view is retained. The fixed predicted-risk
rule remains2%, and causal decisions are frozen before outcome readout.
All comparison components share the CV denominator. Confidence intervals use
3,000 paired locality resamples, conditional on the fitted models and unadjusted
for dependent comparisons. This is not independent final testing.

## Results and Interpretation

The all-event neural controller improves full-policy ADE over its hurdle control
in7/9 point contrasts, but the fixed-count ordering comparison has only1 positive
and5 negative intervals at the control count. At its own count it has1 positive
and3 negative intervals. Direct ranking supervision therefore does not establish
the intended consistent ordering improvement.

All18 all-ADE point contrasts against equally protected damping favor damping;
17 intervals are strictly negative. All18 hard-subset intervals favor damping.
Although all new neural views satisfy the positive-easy percentage limit,12/18
still harm zero-CV rows; the other six contain no zero-CV examples. Damping also
exposes a new tradeoff: the modified easy-event controller fails positive-easy
preservation in three seeds of one fold, with worst degradation4.9464%.

Training-pair support differs substantially: neural/easy receives167,291 valid
pair draws across18,000 updates, compared with2,355,302 for neural/all. These are
repeated training draws, not independent samples. The observed disparity and
realized-ratio/conditional-moment mismatch motivate a fitting-only pairing repair,
not a posthoc deployment threshold or selection of favorable folds.

## Limits

This study changes risk learning, not the trajectory dynamics network. A lower
training ranking loss does not prove calibrated deployment risk. The data are
released detector tracks in image pixels,8 observed/12 predicted rawstride12
positions, not historical raw-frame t50,seconds,meters,human gold or physical
safety. Reserved roles remain closed; no Stage37 recertification,Stage5C,SMC,
true-3D,foundation or submission-ready claim follows.
