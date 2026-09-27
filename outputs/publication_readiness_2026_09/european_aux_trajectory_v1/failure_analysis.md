# Failure Analysis: Task Information Is Not Cost Improvement

## Supported Findings

- Reconstruction failure is not the observed problem: every final parameter,
  optimizer and sampling state matches the original. Instrumentation did not
  accidentally change the trained experiment.
- Cost deficits are present at the first recorded time, not only the final
  state. Both feature families have negative cost4 point contrasts against
  cost-only in all six assignments across the five times.
- True-event supervision can carry information relative to shuffled labels
  without beating no auxiliary supervision. The early easy-harm contrast
  demonstrates this distinction; it cannot be promoted as downstream success.
- The affected severity strata change with assignment and training time.
  A single explanation that every failure comes only from the most extreme
  tail, or only from zero-harm rows, is inconsistent with the error accounting.
- Full and motion inputs have different gradient-conflict patterns. The
  full-input shared cosine is usually positive even while cost learning is
  worse. Raw gradient sign alone is not a sufficient repair criterion.

## Unresolved Causes

Shared-representation capacity, objective scaling, class imbalance, intercept
initialization, optimizer path and continuous-target transport may interact.
The present measurements do not isolate any of them causally. The fitted
event BCE falls in all six full-input assignments from200 to2000, yet this
does not yield consistent cost improvement. Lower classification loss is
not the deployment objective.

The post-hoc prior mismatch is a concrete implementation hypothesis: a rare
cap-event task inherits the substantially larger easy-membership intercept.
This introduces an avoidable constant-prediction BCE penalty under fitting
weights. It may affect the early shared update, but measured gradient sizes
do not prove that explanation. Its repair requires matched actual training.

## What Was Not Done

No checkpoint or threshold was chosen from the plotted curves. No outer
outcome was newly evaluated; no independent selection/calibration/confirmation
role was opened. There was no new forecast model, projection sweep, deployment
change, Stage5C execution or SMC. Source fitting diagnostics cannot establish
generalization, physical safety or a calibrated population-risk guarantee.

Earlier ordinary harm-only, severity-weighting and final-state projection
failures remain part of the record. This study adds timing, not permission
to relabel an already rejected generic repair as a new idea.
