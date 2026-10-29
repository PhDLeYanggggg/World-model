# Negative Mechanism Result: Range Is Not Sufficient

We examined whether an overconfident anchor limits a protected risk head's
ability to learn rare intervention harm. In a frozen-policy diagnostic, the
bounded log-odds branch could not reproduce a substantial fraction of realized
harm on its own selected cohort. This does not imply that the conditional mean
lies outside the architecture's range, but motivates a controlled repair.

We replaced the bounded odds adjustment with a full-range additive probability
adjustment while preserving the causal envelope, nested easy-harm bound and
frozen benefit/reference outputs. Eighteen continuations used identical initial
parameters, source/query draws, optimizers and2,000-step budgets to the previous
isolated control. All six prior controls were retained, and all models were
frozen before the complete recording-level readout.

The wider head improved the matched-intervention training contrast by0.488
percentage points of conservative utility(95% locality-bootstrap CI0.122 to
0.951), but not the corresponding held contrast:0.243(-0.057 to0.591). Held
signed-score MSE increased by2.727(0.629 to5.195). Known easy-risk violations
were44/72, with no view satisfying complete policy support. The apparent
reduction from47 known violations does not establish better protection because
undefined easy-risk views increased from4 to12. The fixed advancement rule failed.

This finding limits a simple representational explanation. Range expansion
also changed optimization: logged median gradient clipping factors were0.059
versus0.506, despite identical optimizer settings. Consequently, these results
do not distinguish a conditional-information limitation from parameterization
conditioning or remaining frozen-reference constraints. They do show that
wider bounded corrections alone are insufficient for safe selected intervention.

These are posthoc mechanism experiments on previously exposed original TRAIN
recordings, not independent main results. Bootstrap uses12 locality groups;
context/seed views are dependent. The three seeds train cost heads rather than
new forecasting producers. Conservative utility is not ADE/FDE, and no world
dynamics, physical safety or foundation claim follows. The result belongs in a
failure/mechanism section, not as the paper's main performance contribution.
