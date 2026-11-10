# OpenDD: rank utility after matching risk feasibility

Reuse the seven OpenDD development locations, three seeds, 8/12 steps,
donor-only cached preference plans, forecast paths, nested menu and
unchanged 2% recording rule. No changes to evaluation, deployment or
external roles. Hold each location out of both fitting and pair creation.

Starting from each archived pairwise head, continue both arms for 300
updates, with identical first minibatches of donor preference pairs.
The continuation control draws a second minibatch of the same kind.
The utility arm replaces that second minibatch with pairs of risk-passing
plans from the same donor recording. Start from a verified winning plan,
test donor alternatives and one-edit moves to another query, and match the
edit count, movement cap and chronological credits. Keep at most one low
gain and one near-winner neighbor per anchor, plus valid cached negatives.
The original trajectory scorer verifies both plans, and the larger minimum
conservative gain over CV and OLS4 wins. If a fitting cell has no such pairs,
stop rather than silently substituting a different objective.

Compare both continuations, the unchanged pairwise checkpoint and the
frozen count scores on the same realized counts and movement caps. Select
plans by predicted scores only; compare complete outputs with the original
scorer. The primary contrast is utility continuation minus ordinary
continuation on dual passes, after seed averaging and equal location
weighting, with 3,000 paired location draws. Also report paired conservative
gain, risk passes and undefined risk. Full-sequence score matching is an
offline diagnostic, not a streaming policy. Unresolved plans retain the
archived matched comparator and remain counted. No external outcome access,
threshold search, or release.
