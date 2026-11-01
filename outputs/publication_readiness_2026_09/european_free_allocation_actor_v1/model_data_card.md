# Free Allocation Actor: Model And Data Card

## Purpose

Study whether a separately parameterized routing actor can improve seven-action
forecast selection without distorting continuous benefit/harm/reference moments.
This is an exposed SOURCE decision-learning experiment,not a new dynamics model
or a deployment certificate. The registered primary and negative results must be
read in `readout.md`;this card does not promote a favorable secondary endpoint.

## Model

Thirty-six independently fitted small Torch actors:two context arms,six SOURCE
contexts,three critic/actor seeds. Each uses133causal fields per action,a32-unit
hidden layer and4,321parameters. Both arms share preprocessing,initial weights,
training queries and fixed2,000-update budgets. The final layer starts at zero.

Inputs comprise a64-dimensional frozen candidate representation,a64-dimensional
context representation,two selected-set descriptors and three frozen cost
diagnostics. Selected context is produced by the existing causal full-joint
policy,not by oracle-best labels. Inference does not receive future endpoints,
future visibility,ground-truth gain/harm or an oracle action. SOURCE future labels
and missing-label envelopes enter training/evaluation only. Normalization uses
past features,not future support or test statistics.

The actor changes utility ranking,not trajectories or the18frozen cost critics.
Offsets are unbounded routing scores;they are not expected-FDE predictions or
calibrated uncertainty. No ordinary residual,latent generation or SMC is trained.
The actual decoder retains original2%predicted all/easy cost constraints and all
seven original eligible candidates. A failed optimizer retains its feasible
independent parent. Such predicted feasibility does not establish actual safety.

## Training

72,000primary AdamW updates plus4,000exact-full-refit updates;native arm64 Torch,
CPU4/interop1/workers0. A separate400-update pilot checks interruption recovery.
Known SOURCE labels train expected utility and original recording constraints;
unknown rows receive causal pessimistic bounds,not fabricated supervised labels.
Full recording/query multiplier updates occur every100steps. Training is a soft
surrogate rather than differentiation through the hard MILP. Changing multiplier
loss values must not be interpreted as comparable calibrated risk or success.

## Data And Statistical Unit

EuropeanSquares oblique-camera detector-silver tracks in local image coordinates.
Observation8,prediction12,stride12rawframes. All149,510unique target rows,
74recordings and12exposed localities remain. The72context/critic-seed views and
overlapping windows are repeated measurements,not72independent sources or
897,060independent examples. Seeds17/29/43 share forecast producer43.

No reserved DroneCrowd labels,independent calibration or confirmation are opened.
SOURCE fit results are resubstitution,not whole-pipeline OOF. Unknown completion
bounds do not repair missing interior observations. Requested endpoints absent
from source keep full-roster FDE not_estimable;an earlier endpoint is not substituted.

## Safety And Limits

Every original recording/context/seed view must meet the unchanged all/easy
positive-harm/reference and easy-degradation requirements. Empty selected easy
support is not_estimable,not safe0. Count-matched comparisons measure utility;
they cannot certify the full deployment policy. Pooled gains and physical proxy
improvements cannot override individual-record failure.

The model is not true3D,a foundation world model,metric prediction,seconds-level
prediction or human-gold evaluation. It is not SDD raw-frame t+50. Stage5C and SMC
remain disabled. Deployment and submission readiness are unchanged by completion.

## Reproduction Boundary

Code,configs,registration,fit receipts and scalar aggregate metrics are public.
Weights,feature/history/latent stores,raw data and decoder checkpoints remain
private/ignored. Public report regeneration is different from training and from
reoptimizing a time-limited decoder. See `reproduction_zh.md` for exact commands.
