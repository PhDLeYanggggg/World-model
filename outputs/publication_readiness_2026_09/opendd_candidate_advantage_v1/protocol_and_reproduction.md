# Candidate-Conditioned Evaluation

## Question And Controls

Can a causal model predict when a **specific** replacement trajectory
beats CV, rather than only when CV has a large error? The
[CV difficulty control](../opendd_hardness_gate_v1/README.md) determines
the number of replacements in each query. Every active comparison
uses that count and the same frozen ego-only EqMotion prediction for
each candidate row. Motion disagreement is a separate ranking control;
CV and outcome-aware gain are the no-intervention and hindsight controls.
Counts are matched, not the sum of peak deviations after reassigning
recipients. The risk readout is a per-row rank with positive-excess
penalties, not a claim of globally optimized joint risk.

## Data Roles

For outer location (h) and fitting location (s), build the
candidate forecast on (s) with the producer fit to the five
locations other than (h,s). One model for each unordered pair
and seed predicts both excluded sites; one output is used for
each orientation of the nesting. No training-row forecast comes
from a producer fitted to that row's site, and no selector
development includes the outer site. Fit-site easy cuts
also exclude the current training source. The outer trajectory
is the existing verified six-source producer for that held site.

The selector sees the eight observed positions, causal motion
descriptors and the planned candidate minus CV path. The
target is the candidate's per-step lower gain versus CV plus
all-risk, easy-risk and signed-easy risk excesses. Easy
support is an auxiliary label. The relative error and risk
terms are scaled by the past RMS step; missing observations
retain their original conservative debits. The two causal
readouts share a fitted head; one uses gain alone and the
other subtracts fixed penalties on positive predicted excess.
The final label-aware assignment under the original risk rule
is diagnostic only. It does not enter feature, target or
policy choice.

The individual choice is related to
[regression with multi-expert deferral](https://proceedings.mlr.press/v235/mao24d.html).
Here a replacement competes for a fixed query budget and is assessed
with recording-level reference support, asymmetric harm and missing
trajectory steps. We do not claim the classifier implements that
paper's surrogate, or that the original 2% rule is a conformal guarantee.

## Local Runtime

Use the native arm64 environment, four Torch threads and no data
workers. The registered checkpoint identity includes source roles,
seed and experiment hash. Checkpoints save optimizer and sampler state
every 100 updates. The ordinary command resumes; replay requires
completed fits and performs no optimizer update.

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_candidate_advantage.py
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_candidate_advantage.py --replay
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_advantage_capacity.py
```

Private data, predictions and weights stay in the local workspace.
All results are exposed OpenDD development, native 8/12 annotation
steps, and dataset-local units. The original 2% rule and 15/30
DroneCrowd split remain unchanged. No Stage5C, SMC, metric claim or
independent readout.
