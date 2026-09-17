# Fit-Only Stationary-Start Identifiability Probe

## Material Passport

Scope: prospective diagnostic within the existing v7 fit recordings only.
Parent task, splits, forecasters, metric, risk ceiling and deployment are unchanged.
Development, calibration and confirmation labels must not be opened. This is
not a new primary endpoint, stationary subset benchmark, or proposal to drop rows.

## Question

The completed bound audit found 365 exactly stationary observed windows,
accounting for 73.3% of pooled fit CV error. Can causal neighbor context distinguish
future annotation change from continued identical positions on those windows?
First establish source identity, unique tracks/runs and label support. Do not
assume an annotation change proves physical motion or that 365 windows are IID.

## Fixed Diagnostic

- Same registered fit windows: eight observed points, twelve requested native
  annotation steps, complete targets. Recompute all counts from current sources.
- Stationary means exactly identical observed x/y across all eight points.
- Label: any requested future x/y differs from current x/y. Report continuous
  future displacement/path and first change step as labels only. No new physical
  displacement tolerance is invented for unverified coordinate units.
- Verify cleaned canonical position rows exactly equal cached rows. Count
  repeated-position runs and unique agents; identify runs beginning at track
  entry, gaps, unavailable neighbor history and single-class folds. Source
  equivalence cannot establish raw sensor-as-of annotation validity.
- Eight-step ego-only features are identical on this subset. A training-only
  smoothed start-frequency prior is therefore the essential control.
- Compare geometry-only context with geometry-plus-motion. Current visible
  neighbor count, complete/aligned support, relative distances, path, speed,
  radial motion, closest-approach proxies and straightness come only from the
  same eight past steps. Normalize using the query's observed positive median
  neighbor distance, with explicit degenerate-support flags. No recording ID,
  absolute location/frame, future mask, endpoint, supplied velocity or learned goal.
- Fixed logistic regression (C=1, max_iter=1000) and ExtraTrees (256 trees,
  min_samples_leaf=10, max_features=1.0, four threads). Seeds 17/29/43, no tuning.
  Fit-only standardization for logistic; no class balancing, probability calibration,
  deployment threshold or new correction head. Save each model and run identity.
- Leave the parent physical-scene folds out in turn. Each model and preprocessing
  excludes its held fold. Empty stationary folds are reported not_run, not filled
  using development data. One-class training uses an explicitly labeled prior
  fallback, not a successful trained classifier.
- Report Brier/log loss, AUROC/AP where defined, held-scene prevalence, train prior,
  seed variation, track counts and stationary-run counts. No window-bootstrap CI
  or independent-generalization claim. All settings retained; no selected winner.
- The comparison diagnoses label predictability, not future direction, trajectory
  improvement, calibrated safety or the irreducible error of all possible models.

## Next Decision

If context has reliable held-fold signal, design a separately registered predictor
that can model stationary-to-moving transitions without broad easy-case drift.
If label support or cross-scene signal fails, report that limitation rather than
adding a neural start head immediately. Independent sites and primary forecasting
evaluation are still required. No Stage5C, SMC, metric or seconds claim.
