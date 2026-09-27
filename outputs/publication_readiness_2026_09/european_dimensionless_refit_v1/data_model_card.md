# Data and Model Boundaries

The new model is the same 88,514-parameter deterministic grouped-history
forecaster, except for the units entering the bounded correction fraction.
It is not a new foundation model, generative rollout, JEPA contribution or
physical-world simulator. Old checkpoints and deployment are untouched.

## Data Roles

All inputs reuse the hash-sealed partial-neighbor observation cache from the
European Squares source-training role. There are 318,969 target queries,
163 recordings and 12 localities. Each producer trains on four localities;
its entire predictor/preprocessing chain excludes its eight readout localities.
Each readout locality has two producer contexts and three seeds. These views
share data and are not independent replications. Source outcomes have been
used for development; their readout is not untouched testing.

No independent selection, calibration or confirmation data are opened. No new
raw download, scene split, endpoint-derived goal, label cleaning or sampling
filter is introduced. Future coordinates and label-validity masks are used
only for supervision and scoring. Poisoning them at inference leaves predictions
unchanged in the synthetic contract check; inherited raw-lineage checks remain
hash-verified, not freshly rerun raw-data conversion.

Histories contain eight requested observations, and outputs twelve future
positions at raw-frame stride12. Up to eight causally nearest agents retain
their observed past support. No future neighbor mask is an input. Coordinates
are image-local detector tracks with silver labels, not human gold. No verified
homography, metric conversion, physical time, true3D or physical safety claim.

## Model and Loss

Within-agent temporal attention and across-agent interaction attention retain
past track association. Inputs are scaled with the same observed-context rule;
the inherited one-unit clamp is not altered. The correction fraction is now
dimensionless, and the outer past-motion radius supplies coordinate units.
The initial output equals the training-selected causal baseline. Correction
remains bounded and zero-budget histories remain at the baseline floor.

Training uses the same native-coordinate objective and training-locality
normalizers, optimizer, schedule and query draws. Removing scale restoration
changes effective gradients/amplitude as well as unit equivariance. A benefit
cannot isolate these mechanisms without additional experiments. The source
loss is not an independently calibrated relative-harm objective.

## Risk and Access

Current accuracy gates are exploratory predictor screens, not authorization
to change deployment. Easy error relative to CV, tails, adverse localities and
zero-reference costs must remain visible even if the matched contrast passes.
No guarantees for unsupported agents/domains or coordinate scales below the
conditioning clamp. Scene-joint consistency and calibrated selection are still
unproven. Private caches/checkpoints/row forecasts stay out of Git; public files
contain only code, aggregate evidence and artifact hashes. Stage5C and SMC stay off.
