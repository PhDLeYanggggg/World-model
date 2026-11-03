# SOURCE Label-Support Model And Data Card

## Intended Use

Diagnose whether missing future labels in the released EuropeanSquares tracks
can be anticipated from causal history on other SOURCE localities. Not physical
visibility, human annotation quality, trajectory forecasting or a deployable
safety filter. Current policy is unchanged.

## Fitting

Eighteen ExtraTrees classifiers: two feature arms, three original producer folds
and actual seeds17/29/43. Each fits four SOURCE localities and predicts the
excluded eight. All128trees finish; depth12,minleaf64,maxfeatures1/3. Equal-site
sample weights and mean/std are fitted on training sites only, clip8. No class
reweighting or held-label threshold/calibration selection. Geometry has476past
fields; the second arm adds64past detector-history fields. Future-valid masks
are supervised labels only, not feature columns or normalizer inputs.

Positive label: all12requested future labels absent. Partial futures remain
negative for this particular target. This event concerns the release/tracker,
not known physical occlusion. Static future masks and eventual track lengths
are not available to inference. No future endpoint, central velocity, test goal
or held normalization is used. Input provenance is cached_verified from the
existing SOURCE feature conversion/role checks, not a new full raw-data audit.

Data are the original149,510-window whitelist at74recordings/twelve SOURCE
localities, a subset of the earlier318,969-window forecaster cohort. Windows
overlap; repeated fold/seed views are not independent people. obs8,pred12,
rawstride12,image-local pixels, released detector/tracker silver. No metric,
seconds, human-gold, true3D or foundation claim.

## Runtime And Artifacts

Actual full fits:2304primary trees,128first-full-refit trees,128pilot trees.
Training267.972s,4.513GBpeakRSS on native arm64. Training uses four threads;
prediction reduction is single-thread for exact replay, workers0. Checkpoint
every32trees, explicit same-binding resume. Real32->64/direct64 and first full
refit tree states/predictions are exact. All18held probability arrays and144
model/locality groups replay exactly. These are forests, not new Torch dynamics
and not NumPy arithmetic presented as a neural model.

Weights and prediction arrays remain in ignored
`data/stage_cvpr2027_experiments/european_source_label_support_v1/`.
Public fits/protocol/registration/group scalars bind data roles, software and
weight identities. Private inputs/checkpoints still limit anonymous cold start.

## Results And Limits

Geometry+detector AUROC0.706249 [0.648098,0.750792], AUPRC0.144023; registered
AUROC feature effect+0.103922 [0.059967,0.162186]. Intervals use3000locality
resamples after averaging excluded folds and actual seeds. SOURCE is exposed
development; positive component evidence is not independent confirmation.

Calibration shifts, locality020 remains below chance, and the critical082
recording can have much larger unknown prevalence than its predicted scores.
No selective policy, complete2%risk pass, physical safety or submission
readiness follows. New bound-target construction is unit tested only; its
critic training is not_run. No private calibration/final/DroneCrowd arrays
opened. No Stage5C execution or SMC.
