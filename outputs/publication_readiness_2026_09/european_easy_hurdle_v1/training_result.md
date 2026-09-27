# Paired Risk-Head Training

## Completed, Not An Efficacy Result

All108registered fitting groups and216risk heads completed2,000updates each,
432,000cumulative model updates. This is fresh real fitting on detector-silver
trajectory labels, not a NumPy fallback or a synthetic runtime probe. Forecasts,
floor, utility, all-risk and the source-excluded input lineage remain
cached_verified. Held-development action evaluation is not_run; independent
selection/calibration/confirmation remain closed.

The two heads share architecture, initialization, preprocessing and exact
source-balanced query draws. Only their registered loss differs: marginal
easy-risk supervision versus marginal plus occurrence/conditional-cost
supervision. Every head's own fixed fitting-monitor objective is lower at the
end than at initialization. This is a training observation, not downstream
lift, calibrated probability, a risk certificate or a deployment recommendation.

## Execution Evidence

| Check | Actual result |
|---|---|
| Input transfer | 108hash-verified fitting-only packets,4,909,669,199bytes |
| Real pilot | job37563466;2heads x100updates;9.3602s fitting;34s scheduler |
| Full fit | job37563607;COMPLETED0:0;24min32s scheduler;1,466.504s runner |
| Full-fit peak RSS | 6,222,164KiB,about5.93GiB |
| Final checkpoints | 216files,89,517,817bytes,all SHA-256 rechecked |
| Environment | Linux native x86_64,Python3.11.6,Torch2.12.0+cpu,NumPy2.4.6 |
| Resources | one task,CPU4,interop1,workers0,16GiB allocation |
| Model budget | 432,000includes200pilot updates resumed;not432,000additional updates |
| Verification | job37564122;COMPLETED0:0;35s scheduler;29.953s runner |
| Verification peak RSS | 2,051,380KiB |
| First-pair full replay | 2heads,4,000extra updates;every state field exact except elapsed time |
| Other214heads | checkpoint/hash/source/finite-value/budget/paired-sampling checks,not independent retraining |
| Local scoped tests | 42passed,8files;full legacy integration suite not_run |

The full-fit record is [create_training_freeze.json](create_training_freeze.json),
SHA-256`57412b73dafe2d3abf408f7655c4c0818c90a8befef427f7712d11b16634c821`.
The allocated-node audit and replay record is
[create_training_verification.json](create_training_verification.json),
SHA-256`8028bf02af4e43eb992f614a79a8953e18d888dbf00351cd6922d50c0d0894cf`.
This establishes same-runtime replay, not bitwise equivalence to Apple Silicon.

## Failures Retained

The local pilot was refused before its first update because the unchanged10GiB
reserve was unavailable. No old data were deleted and no training scope was
reduced. The first remote runtime job failed before Torch because a non-login
shell lacked module initialization. A separate repaired environment passed an
actual optimizer/resume probe before real fitting.

Intermittent SSH failures interrupted upload and two submission attempts.
Completed packets were rehashed; remaining packets used one temporary serial
connection. The absence of remote submission intent, receipt, script and queued
job was checked before the successful pilot submission. No duplicate training
job or persistent authentication/configuration change was made. The simulation
project, environment and jobs were not reused or modified.

## Next Registered Step

Keep the model hashes frozen. Build causal-only action packets and run the
registered raw/marginal/supervised independent, joint and common-count matched
controls. Freeze all actions before reading held-development costs. Then report
paired locality uncertainty, retained benefit, harm, easy preservation and
undefined selected-risk denominators without threshold tuning. No claim of
improvement is available until that readout and its verification complete.

Bulk checkpoints and arrays remain private on CREATE because the local disk
reserve is still binding. The remote action/readout transport is the next
implementation step, not something completed by this fitting audit.

Scope remains12opened development localities,obs8/pred12,stride12raw frames,
image-local detector silver. The original incomplete primary is not replaced.
No metric/seconds,human-gold,physical-safety,true3D,foundation or submission-ready
claim. Deployment is unchanged;Stage5C andSMC remain disabled.
