# Rank the recipient by easy-reference support

This is a targeted development follow-up to the factorized head. In that
study, 15 of its 20 held releases had no easy denominator. Keep the
seven existing OpenDD vehicle locations, 8/12 native steps, three seeds,
the same queries, forecast bank, radius and original 2% limits. Do not
open Songdo or any reserved confirmation labels.

For each of the 63 double exclusions (21 unordered site pairs, three
seeds), reuse the matching vehicle predictor and factorized value head.
Fit one binary support model on one row per observed donor actor-window,
without either the held or the calibration site. The positive training
label is a nonzero available-step CV error at or below the already
frozen donor easy cutoff. It is defined once per actor, not once per
candidate path. Use only recent displacement, speed and past-prefix
CV backtest errors as features; express absolute sizes in units of the
donor cutoff and compress them with log1p. Fit 200 AdamW updates per head,
with the existing donor-balanced batch sampler and 100-step checkpoints.

Compare three fixed-count arms at the same query and radius:

1. The archived factorized value choice.
2. Pick the actor with the highest learned support score; within each
   actor take its best legal factorized-value candidate.
3. Same recipient rule using the reciprocal of one plus the worst
   cutoff-scaled CV backtest error as the support score. This is a
   non-trained prefix control.

In arms 2/3 break ties by candidate value, then stable actor index. Rank
their release scores by support times the positive part of the selected
candidate value. If the value is nonpositive, the candidate is still
counted in the matched diagnostic but never eligible for release. The
first arm retains its original release score and threshold.

Both directions use the original score-prefix calibration with ties
grouped and the unchanged all/easy/signed risk checks and positive
conservative gains against CV and OLS4. Score the held directions only
after freezing choices and thresholds. Report support, actual risk and
two-reference utility separately. Average across directions and seeds
within site before paired site bootstrap (3,000 draws). This is a
one-edit mechanism test on reused development sites, not a final
evaluation, joint-policy comparison or statistical safety guarantee.
