# OpenDD: predict the value of each legal edit count

Keep the seven OpenDD development locations, three seeds, frozen forecast
paths, frozen nested nominations, 8 observation steps, 12 prediction steps
and the original recording-level scorer. For each outer location and seed,
use the six other locations for training. Produce each donor menu with the
archived ranker fitted without that donor or outer location. Derive the donor
easy cutoff from the five remaining locations and the outer cutoff from its
six donors. Do not tune on the outer outcomes.

Fit a small prefix-value network on past-only features. For each legal count
one or two it estimates the conservative gain over CV, incremental gain over
OLS4 (subtract the CV fallback contribution), and the three signed terms of
the original risk constraints. Train with Smooth L1, equal donor sampling,
500 updates and no validation-driven hyperparameter search. At inference take
the smaller predicted gain and subtract the largest positive predicted risk
term. All terms are in comparable units after reversing normalization. This
fixed penalty is a score, not a risk guarantee. The zero-edit value is zero.

Compare the same-sized gain-only fit, the five-output fit with its penalty
disabled, and the five-output fit with the penalty, against the frozen count
head. Use the unchanged causal token decoder. Then evaluate all score heads
with the frozen online controller's exact per-view count and no more summed
peak displacement using the label-blind offline allocator. Full-sequence
score matching is not online. Report whole-roster original risk and dual
pass counts, equal-location paired conservative gain and location-bootstrap
intervals, not just results on edited views. No external confirmation labels
or thresholds are used. A development improvement does not authorize release.
