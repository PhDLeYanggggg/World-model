# Does Observed Image Motion Predict a Recorded Start?

## Fixed Question and Scope

The past-frame experiment repairs coordinate consistency but all fresh trajectory
fits remain below CV. Before another residual network, isolate whether existing
observed motion summaries predict any annotated movement after a static history.
This is a fit-only information probe, not a replacement forecasting endpoint,
independent confirmation, deployment gate or architectural novelty claim.

Keep all 365 previously audited stationary windows: 81 ETH, 284 Hotel, 31 local
agent IDs and 45 stationary runs. Hold out each entire physical scene. Zara has
no stationary rows and is explicitly not_run. The full approved 11,966-window
8-to-12 forecasting cohort and primary metric remain untouched. No new SDD
admission, Students, development, calibration or confirmation access.

## Controlled Comparisons

Join the frozen stationary-neighbor cache to the verified full-cohort motion
cache by recording, agent and frame. Check every selected row, fold, scene and
label against the independently stored full-cohort arrays. Feature construction
accepts only past summaries; future labels, source run boundaries and identifiers
are not estimator inputs. Offline annotated histories retain their retrospective
interpolation limitation; optical flow is not verified physical body motion.

Four fixed inputs: pooled neighbors (13); add image quality/support (48); add
motion magnitudes (76); add directed motion (118). These test added information,
not equal parameter counts. Existing motion axes/scales are not changed. Quality
is a required control so camera/support signals are not mistaken for motion lift.

Logistic regression: train-only StandardScaler, C=1, max_iter=1000. ExtraTrees:
256 trees, min_samples_leaf=10, max_features=1, single-process n_jobs=1. Seeds
17/29/43; two supported folds and four inputs give 48 fresh classifier fits.
Logistic seeds are numerical replications, not independent experiments. No
class reweighting, feature tuning, probability threshold search or held-label
calibration. Compare with the smoothed training-only change-rate prior.

The label is any nonzero annotated displacement within twelve requested future
steps, not verified intention. Proper probability scores (Brier/log loss), AUROC
and AUPRC are diagnostic. Report training versus held fit, each direction, each
seed and agent/run-balanced scores. Compare magnitude/directed to quality on
identical rows, and retain absolute differences. Compute 2,000 paired held-agent
bootstrap draws on seed-mean losses, conditional on these fitted models/sites;
five ETH agents do not justify an independent-scene or formal risk claim.

## Execution and Decision

Checkpoint each completed classifier atomically, record PID/heartbeat and persist
probabilities privately. Resume verifies hashes; full saved-model prediction
replay is required. The inputs are small; native arm64 local execution suffices,
without another runtime or unchanged remote-access probe. No Torch training is
claimed for sklearn fits. No raw arrays, weights or media are committed.

A useful signal requires improvement over both the training prior and quality
control in both directions, with seed/group results disclosed. Even that would
not prove displacement/direction accuracy or authorize deployment. Failure means
do not launch a residual head from these summaries or tune another threshold.
Seek different observed state-change information or separately authorized
independent support. No primary change, Stage5C, SMC or metric/seconds claim.
