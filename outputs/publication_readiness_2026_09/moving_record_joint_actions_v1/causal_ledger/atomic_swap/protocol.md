# Count-Preserving Joint Forecast Exchange

Test whether a learned insertion and withdrawal can work together
without increasing the number of switched agents. Use the same
native observation-8, prediction-12 own-B development recordings,
frozen forecasts, direct policy and source roles as the previous
rescue experiment. Held outcomes here are exposed development,
not new independent validation.

An atomic pair inserts one nonidentical forecast on an unselected
agent and restores the motion baseline on a different, selected
agent of the same current query. No future query is visible. Permit
at most one pair per recording. This keeps every query's original
intervention count, including zero. The action bank and risk
requirements are unchanged; this is not the broader ordered
insertion/deletion action family.

Before fitting, exhaustively enumerate the pair family and
verify one witness per recoverable recording with the original
scorer. A positive teacher label is a new recording pass with
nonnegative paired lower utility. An unknown removal still pays
the original disagreement radius in the paired bound. Publish
capacity counts only. The capacity phase does not select or
change models, features, thresholds, or data roles.

Fit a 24-dimensional per-side network with past motion and
predicted costs. Learn an additive pair logit and an interaction
logit using a product of the two embeddings. Both outputs use
identical candidate pairs, rescue targets, mini-batches and
side encoders. Fit 220 updates per fold, batch 256. Upstream
features are source-excluded within fitting. Outcomes and
recording totals are used only for supervision, calibration and
scoring, never as features.

At each query each arm ranks one pair and can take the first
one exceeding its calibration cutoff. Calibrate on the existing
six donor sources only: require additional passes, no lost
passes, no new signed-easy violations and nonnegative paired
lower utility. If no cutoff qualifies, keep the direct policy.
The primary matched comparison probes one pair per arm in the
first legal query (no probe if no legal pair). Report the
paired source bootstrap with 3,000 draws, whole-record scores,
counts and missing references separately. Do not use empty
probes as safety evidence.

Keep the original 2% risk and support rules and report all
102 repeated views across 14 source groups and three forecast
seeds. No new independent outcome, deployable policy, metric,
seconds-level, Stage5C or SMC claim follows.
