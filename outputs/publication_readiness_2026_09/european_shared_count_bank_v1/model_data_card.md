# Model and data card: common-count bank control

## Intended question

Compare fixed six-physical and seven-physical/neural candidate banks after both
receive current-query optimization at a common feasible intervention count.
This is an exposed SOURCE fitting experiment, not an independent benchmark or
a newly trained world model. Deployment is unchanged.

## Components

- Six physical trajectories: position, causal finite-difference velocity,
  damping0.90/0.97, history OLS4/OLS8 velocity.
- One previously trained history forecaster, producer seed43, unchanged.
- Eighteen cached_verified cost heads: six context partitions, seeds17/29/43.
- Four fresh decoders at the physical independent rule's per-query count.
- Two previously frozen seven-action controls, freshly rescored; their old
  joint choices are cached_verified, not solved again in this experiment.
- Zero new model fits or gradient updates.

## Data and labels

European-source image-local detector-silver trajectories use eight observed
and twelve requested future steps at raw stride12. The same149,510 unique
target queries,74 recordings and12 localities generate repeated context/seed
views. These views and overlapping windows are not independent observations.
Future ADE/FDE, easy and hard labels enter scoring only, not allocation inputs.

Inputs are past-only causal features, frozen candidate paths and moment
predictions, causal action availability, recording/frame keys and canonical row
IDs. No future endpoint, central velocity, test-derived goals or normalization.
The set of query targets need not be every visible agent.

Unknown outcomes stay in completion accounting. Missing interior future labels
and wholly unknown rows are different limitations. Full-roster FDE cannot be
reported when a required endpoint is absent. No unknown error is filled with
zero; an empty selected-risk denominator is not a successful safety estimate.

## Selection and reproducibility

The scientific code, configuration, count rule, outcomes and protocol were
committed before the pilot/readout. There is no outcome-based checkpoint,
threshold, candidate or cohort selection in this experiment. Separate full
replay checks every fresh prediction, four action hashes, legacy score and
summary; interruption recovery is tested before full execution.

Native arm64 CPU4/interop1/workers0, atomic private progress, heartbeat and
explicit resume are used. A 2s/query solver boundary returns a checked feasible
anchor on nonoptimality or failed postcheck. Such a result is not called exact.
No CREATE job or external confirmation dataset is consumed.

## Limitations and prohibited use

Predicted objective nesting is not observed risk control. Same-count comparisons
are utility controls, not permission to waive any complete-recording2% check.
These are conditional, multiply exposed fitting results; three cost-head seeds
are not three newly trained forecasters. No simultaneous or population guarantee.
No pairwise interaction dynamics is optimized, and no physical safety is verified.
No independent confirmation, submission readiness or deployment promotion follows.

Coordinates are not verified metric or time-calibrated. This is not true3D,
foundation training, human-gold annotation, Stage5C execution or SMC.
