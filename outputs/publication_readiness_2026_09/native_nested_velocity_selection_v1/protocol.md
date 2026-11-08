# Can causal features recover the velocity forecast headroom?

Freeze the masked-symmetric velocity candidate before training the
selector. Use the original 69,746 fitting windows, 8 observed and
12 forecast annotation steps, three seeds, and nine source groups.
For each outer held source, use the cached forecast trained on the
other eight groups. For selector training, generate each inner row's
candidate with a fresh forecaster trained on the seven groups that
exclude both the outer held source and that row's source. Reuse each
unordered donor fit in both valid orientations; do not train a
selector on ordinary one-level out-of-fold predictions.

Train two matched regressors from ego history, near-peer history
summary and fixed causal candidate geometry. One predicts the
full-grid lower utility; the other predicts the same utility and
upper positive harm and subtracts expected harm from its score.
Missing-step loss is bounded by prediction disagreement. No future
label, visibility mask or endpoint goes into inference. Train 800
updates per donor, 500 per selector. No epochs or thresholds chosen
on the reported sources.

Compare exactly one nonidentity edit per eligible multi-agent query
across both learned arms, deterministic random, smallest displacement
and hindsight. Include no edit (OLS4) and CV as anchors. Report paired
full-grid lower scores, conservative direct comparisons, switch counts,
positive selected counts, hard contributions and known easy positive
harm (not an official safety test). A strict predicted-score-above-zero
option is a diagnostic with different counts, not a certified policy.
Resample nine source groups 3,000 times after averaging three seeds.
Hindsight is a capacity ceiling, not a real policy. No independent
evaluation, change to the 15/30 split or original 2% rule, deployment,
Stage5C, SMC, metric or seconds claim.
