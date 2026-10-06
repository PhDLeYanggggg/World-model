# Complete Fixed-Final TRAIN Run

## Verified Execution

Observed 2026-10-06 09:08:45 UTC. All four tasks of array 37799286 and the
complete-array join 37799287 finished with exit 0:0. The queue no longer lists
these jobs; successful accounting and artifacts, not queue disappearance,
establish completion. The join checked every final checkpoint checksum.

- 24 TRAIN packets, 72 source/head-seed identities, 216 fits.
- Three arms: no auxiliary task, row-mean auxiliary control, temporal auxiliary.
- Seeds 17, 29, 43: 72 fits each; 12 exposed development localities.
- Every fit reached its registered 2000 updates: 432000 accumulated formal
  optimizer steps. Pilot steps are included in those totals, not added again.
- All 216 metadata records and checkpoint hashes verified; final checkpoint
  payloads total 48433116 bytes. Weights remain in the owned CREATE directory;
  the local collector saves light metadata only.
- Each trace contains step 0, step 1 and 100-step checkpoints through step 2000;
  all recorded monitor values, losses and gradient norms are finite.
- The 156 related execution tests passed before submission; the original 127
  scientific code/config bindings remain unchanged.

Task elapsed times were 3:42:21, 0:06:11, 0:06:11 and 0:05:17. The join took
24 seconds. The large timing difference is observed, not explained by this
audit; no runtime-based fit selection or reduced budgets occurred. The longest
task stayed below 12 hours. The original 43.45-hour serial estimate remains an
estimate with its failed single-job feasibility gate preserved.

## Training Monitors

Unweighted means across 72 head fits per arm, fixed TRAIN monitor, step 0 to
step 2000. These are not validation losses, raw FDE or independent observations.

| Arm | Primary initial | Primary final | Fits with primary decrease | Auxiliary initial | Auxiliary final |
|---|---:|---:|---:|---:|---:|
| No auxiliary | 0.863542 | 0.362193 | 72/72 | 3.139850 | 3.139850 |
| Row-mean auxiliary | 0.863542 | 0.405558 | 70/72 | 2.182813 | 0.807527 |
| Temporal auxiliary | 0.863542 | 0.366840 | 72/72 | 3.139850 | 1.111527 |

Training is finite and complete. Temporal auxiliary loss decreases, but its mean
primary TRAIN loss does not beat the no-auxiliary arm. Neither fact determines
selected-policy utility or safety. No hyperparameters or thresholds were changed
after these monitors were read.

## Admission and Limits

Result source: `fresh_run` CREATE training; `cached_verified` checksum-bound
collection and frozen preprocessing. The complete fixed-final readout admission
now passes for all 216 heads. New development predictions and the seven-arm
comparison were `not_run` at this freeze. The freeze and inventory are committed
before opening that readout. All registered controls, paired completion bounds,
2% risk definition and 3000 locality bootstrap draws remain unchanged.

These are cost-head experiments around frozen forecasters, not newly trained
world-dynamics forecasters. This is exposed source-development evidence, not
independent confirmation or a deployment promotion. Independent calibration and
confirmation remain closed; Stage5C and SMC remain off. No metric, seconds-level,
true-3D or foundation-model claim follows from this run.
