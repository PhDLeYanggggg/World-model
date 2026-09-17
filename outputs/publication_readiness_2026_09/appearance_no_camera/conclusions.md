# Camera-Input Removal Is Not a Cross-Scene Repair

## Material Passport

`fresh_run`: six real PyTorch refits of the eight-past-RGB predictor, two held
fit-scene directions and seeds17/29/43, each at1,000 updates. Only four learned
camera-Jacobian input dimensions are zeroed; the original28 geometry/neighbor
features, images, masks, initialization, sampler, loss, training normalization,
output coordinate mapping and0.9 diagnostic gate are unchanged. Total6,000
updates and109.87s summed CPU fit time. This is a small diagnostic, not medium.

`cached_verified`: all six final checkpoints reproduce saved trajectories,
probabilities and guarded outputs exactly. Resume verifies six completed models
with zero new training/evaluation. PID67849/session1153 exited0; no active job
from this experiment. Source code/config/checkpoints and original result hashes
remain bound. No original model, target or report is overwritten.

## Complete Matched Result

Three-seed guarded ADE improvement versus exactzero/CV on the stationary subset:

| Held scene | Original past-RGB | No-camera refit | New unrestricted gain | New switch rate |
| --- | ---: | ---: | ---: | ---: |
| ETH | -14.98% | -19.64% | -18.95% | 65.84% |
| Hotel | -172.63% | -86.28% | -397.45% | 22.07% |

All six refitted guarded results are negative. Hotel damage falls, but ETH
worsens. Hotel unrestricted predictions are nearly as bad as before
(-401.80% original versus -397.45% refit); less guarded damage must not be called
a generally stronger dynamics model. Start Brier lift remains negative on both
held sites. Every seed, raw ADE/FDE, losses and easy harm are in
[metrics.json](metrics.json); [replay.json](replay.json) contains verified means.

Native coordinate units are unverified. Still/easy CV error is exactlyzero,
making a percentage easy-degradation gate undefined; absolute error is retained.
The6,000updates completed without nonfinite loss or shortened budgets. Runtime
success does not rescue failed predictive results.

## What the Combined Evidence Changes

The prior feature-box experiment and this actual refit reject the simple repair
hypothesis that a camera-range fix alone recovers useful cross-scene trajectories.
They do not prove that camera information, visual history or all neural models
are useless. Only31stationary agents are represented, with five in ETH, and
the two sites differ strongly. The classifier also learns change probability,
not the utility of a particular directional prediction. These remain distinct
data-support and objective limitations.

The next work should broaden verified fit-context support, not repeat thresholds
or camera-normalizer variants on these same windows. The approved fit roster
already containsETH/Hotel andZara01/02/03. LocalZara01/02 videos and matrix files
exist; their past-frame/coordinate mapping and source-use scope still need
checking before input admission. They are one physicalZara scene, not two new
independent domains. Zara03 has no localvideo in the inspected directory and
retains a packaged-population limitation. File presence is not modality admission
or new confirmation evidence. DevelopmentStudents recordings remain unopened
by these probes.

Preserve the eight-observed/twelve-predicted task and raw50 supplement. The
prospective primary-metric decision is still pending; no retrospective metric
replacement or relabeling of exposed fit data. Independent calibration and
confirmation support remain necessary for publication claims. The clean parent
comparison still uses CV; historicalStage37 scores remain exploratory under
lineage review. No new deployment, true3D/foundation result, metric/seconds
claim, Stage5C execution or SMC. The overall research goal is incomplete.
