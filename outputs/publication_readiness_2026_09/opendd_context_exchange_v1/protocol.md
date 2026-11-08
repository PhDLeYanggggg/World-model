# Past Scene Context for Exact Support-Preserving Exchange

Test whether real, time-aligned scene motion supplies decision signal
that the previous own-history pair models did not have. Use the same
OpenDD 8/12 development roster, seven held locations, three seeds,
fixed-half reference, original 2% readout, and exact one-transfer menu.
The frozen incumbent direction, pair-excluded donor forecasts,
evaluation recipients and realized query radius are unchanged.

Reuse the previous verified eight-timestamp context caches. Check the
pedestrian rows against current past-only neighbor extraction. Use
published historical positions only: past relative range, bearing,
observed relative motion and short constant-relative-velocity approach
features. Explicit validity bits distinguish unavailable input from
zero motion. No future label, released velocity, or image is a feature.

Refit four arms with equal-sized input and the same exact pair labels,
site/query/action weighting and 24-tree configuration:

1. \`own\`: the existing exact-action feature set.
2. \`pedestrian\`: add up to two past pedestrian neighbors per recipient.
3. \`traffic\` (primary context arm): add up to two motor-traffic histories.
4. \`shuffled_traffic\`: preserve the pedestrian features and exchange
   actor-frame traffic summaries cyclically within the same current query.
   This preserves query-level traffic distribution but changes attribution.

Disabled feature slots are zero-filled. All arms use the existing
positive predicted lower gain and nonpositive predicted risk-numerator
changes, with no threshold search. Evaluate full policies against fixed
half and use direct paired lower comparisons between the arms; include
original risk pass transitions and actual choice quality. Report the
past-defined motor-support slice only as descriptive. Average seeds,
then use 3,000 paired location bootstrap draws; no window-level
independence claim.

To support a traffic contribution, the primary arm must improve on
the pedestrian and shuffled controls by lower confidence endpoint,
and not reduce original passes vs fixed half. The original full
recording requirement is unchanged. No external evaluation, confirmation
access, deployment, Stage5C, or SMC.
