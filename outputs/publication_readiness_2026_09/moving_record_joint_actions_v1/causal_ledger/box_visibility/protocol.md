# Observed Box Geometry for Insertion Retrieval

Test whether information absent from center-only trajectories
improves retrieval of insertions that supply easy reference or
repair a recording. Use the full original 125,540-row own-B
roster, 34 recordings, 14 source groups and the three frozen
forecast seeds. This is development, not independent evaluation.

Join admitted released MOT boxes by recording, frame and agent.
Verify annotation hashes, source groups, and exact center equality
at all eight observed steps. For each observed frame compute log
area, log aspect ratio, crowd count, maximum intersection over
target area, overlap count and scale-relative nearest-center
distance. Summarize current, mean, dispersion and first-to-last
change, plus maximum overlap/proximity descriptors, for 27
features. Geometry is not an occlusion annotation. No pre-window
track age, frame-bound estimate, RGB motion or post-query feature
is used. No future target array is opened during feature building.

Compare trajectory/candidate features against the same input plus
box geometry. Use the same eligible action roster and two targets:
observed easy reference and whole-record non-loss rescue. Match
training draws, hidden width 32, positive loss weight, 220 steps,
source folds and donor threshold grid. Upstream predictions and
normalization are fitted without the held source. Rank by the
product of the two learned scores (not calibrated probabilities).
Probe the same first eligible query in both arms. The source-paired
3,000-draw bootstrap reports support and non-loss rescue contrasts.

Both operating arms may add at most one forecast, only at a query
with a spare slot under the original cap. Calibrate on six donor
sources only, requiring new passes, no lost passes, no new signed
easy violation and nonnegative paired lower utility. Score the
frozen choices using the original recording checks, including
undefined reference support and the 2% threshold. A component
gain or fallback is not a policy pass. No metric, seconds-level,
deployment, Stage5C or SMC claim follows.
