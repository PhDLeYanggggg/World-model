# What pushes the local velocity model back to OLS4?

The earlier near-neighbor forecast scarcely moved from its OLS4
initialization. Change only the training loss in a two-by-two design:
debit each missing step by prediction disagreement or mask that step
out; weight positive per-row harm twice or leave it symmetric.
Keep twelve in the loss denominator so changing the mask does not
also reweight partially observed rows upward.

Use the original near architecture, history, bounds, initialization,
eight observed and twelve predicted native steps, and 800 updates.
Verify the cached grid-plus-harm arm and train the other three arms
on three seeds and nine leave-one-fitting-source-out folds. Match
training draws to the existing checkpoints. Freeze the whole-roster
causal predictions before scoring. No model or threshold selection
on the reported folds.

On the same roster, score all four arms against OLS4 and test the
paired effect of each factor at both levels of the other factor.
Use the historical available-label and wholly-unknown-row scores
as diagnostics and the per-missing-step lower score as the primary
full-grid comparison. Also report gain on rows with all twelve
labels, with a separate CV denominator for those rows, and report
median endpoint changes and known easy harm. Complete-label
membership is used only in evaluation and does not identify the
effect on incomplete rows. Resample the nine sources after seed
averaging. The benchmark's 2% rule is unchanged. Use no development,
external calibration or confirmation outcomes and make no new
deployment, metric, seconds, Stage5C or SMC claim.
