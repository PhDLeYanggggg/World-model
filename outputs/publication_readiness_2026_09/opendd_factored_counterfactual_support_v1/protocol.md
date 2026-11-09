# Factorized easy-reference loss

Development follow-up to the observed rare-event failure of direct
whole-record loss prediction. Preserve v153 outer slots, candidate
paths, single-query replacement, count, radii, budget, split lineage
and original evaluation rule. Train one fresh model per site and seed
with the same normalized gain target, feature schema and optimizer.
Train local reference loss on all rows. Train the conditional event
of losing all reference only on rows with local loss, which makes the
whole-loss estimate the product of two probabilities. Both binary
terms have weight 0.2; the conditional mean is over positive local
labels. This changes the training loss as well as the output structure.

Deploy neither arm. Compare the product gate to the same fitted model
using only its local component, and to frozen v153 direct whole,
direct local, and past-prefix controls. Keep the fixed gain margin and
0.20 estimated probability cutoff. Do not select thresholds on these
sites. Include an attribution comparison with the same query clocks
as the product gate, using its gain predictions without a loss gate.
Use three seeds and paired location intervals, all original 2% checks,
and actual lost/restored reference. No independent calibration or
confirmation labels. This follow-up is not a fresh test of v153.
