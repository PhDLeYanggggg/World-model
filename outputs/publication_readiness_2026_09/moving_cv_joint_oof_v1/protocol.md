# Source-Excluded Donor Bank for Causal Choice

The existing A forecasts remain frozen. Their two other producer folds
trained on A sources, so they cannot supply safe donor predictions. Build a
new bank entirely within the nine nonempty approved fitting sources.

- Make three disjoint three-source held groups, assigned by decreasing
  fitting row count to the currently smallest eligible fold. For each seed
  and PCT/EqMotion family, train from scratch on the remaining six sources.
  Every donor candidate is predicted by a model that excluded its entire
  source group, not just the row. No A source enters any fit.
- Same pinned architecture and fixed EqMotion head as the original A bank;
  use OLS4 as the fixed forecaster anchor. Fit-only source-normalized masked
  ADE and matched draws across families. Three seeds, 600 updates per cell,
  batch 16, four native CPU threads, zero workers, checkpoints every 100.
  This fixed-budget training provides donor trajectories, not a saturated
  predictor comparison. The frozen A bank used a longer training schedule.
- Output only a bound checkpoint/forecast receipt in Git. Record source
  exclusions, held row identity, aligned finite predictions, matched draws,
  and complete steps. All outcomes and per-row candidate arrays stay local.
- No A targets are loaded by this producer runner. No new external role,
  independent confirmation, threshold search, deployment, Stage5C or SMC.
