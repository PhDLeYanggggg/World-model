# Does Local Neighbor Motion Add Beyond Pooled Motion?

Use the existing source-separated 69,746-row training cohort and
the three frozen ungated OLS4-relative checkpoints as shared
backbones. Freeze each backbone. Train the same small adapter
with ego features only, ego plus pooled motion of the eight
closest complete-history neighbors, or the pooled features
plus local relative-motion features. Pooled motion is not a
camera estimate. Local features use observed relative position,
relative velocity, velocity centered around the neighbor mean,
and signed closing speed. All masks and values are past-only.

Use matched initialization, source-balanced draw sequences,
loss, optimizer and 900 updates per arm and seed, at batch
size 128. No model selection. Freeze all nine development
forecasts before scoring. The original 14 development sources
are excluded from all training. Keep the exact original gain
recipients and substitution set: if any arm would be identity
to CV within one millionth of an image pixel, all keep the
original chosen path on that row. All query counts, CV fallback,
OLS4 reference and recording 2% and support checks are unchanged.

The primary contrast is local vs pooled at the same recipients.
Report masked ADE conservative utility and relative-path ADE
using the same existing four-nearest-neighbor graph, with the
original available-pair and full-grid lower comparisons. Report
pooled vs ego and each against the original gain policy. Pair
error is not a collision rate. Use 3,000 paired source draws
after averaging the three actual training seeds. Keep all
undefined risk or accuracy views explicit.

This is a test of a specific context adapter on exposed
development, not evidence of independent transfer or physical
safety. No external access, policy promotion, Stage5C or SMC.
