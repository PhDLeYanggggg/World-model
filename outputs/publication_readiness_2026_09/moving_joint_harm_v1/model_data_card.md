# Joint-Harm Model And Data Card

**Status:** complete exposed-SOURCE research experiment; not deployable under
the original contract. No new trajectory dynamics or generative model was
trained. The model is an action-cost critic protecting a frozen forecaster.

## Model

- Two equal-architecture arms: product_moment and event_supervised.
- Each head: width 64, 28,489 parameters, 7 causal amplitude candidates, query-level
  other-agent context, four exclusive event probabilities, three conditional
  fractions and five expected moments B/H/R/ER/EH.
- The matched treatment adds unweighted categorical cross entropy and
  positive-event conditional-fraction MSE. The common loss is normalized
  moment MSE plus 0.5 net-gain MSE. No inference-time oracle or future labels.
- Input: 380 declared causal local-history/action fields; 48 unselected neural
  fields are explicitly zeroed. Fitting-only means/scales, target RMS and bias.
  Candidate rollouts are causal forecasts, not observed future endpoints.
- 108 heads = 18 contexts x 3 inner source parts x 2 arms. Three seeds: 17/29/43.
  These are not 108 independent scenes or 108 independent scientific replications.
- 2,000 updates per head; checkpoint/heartbeat every 100 updates; 4 CPU threads,
  1 interop thread, 0 workers. All 54 paired initializations and samplers agree.

## Data And Labels

The parent admission is 195,286 past-eligible rows from 64 MovingDroneCrowd++
recordings. Only 58 recordings across 23 source groups support the registered
8/12 task. All 276 source/orientation views and 696 record views are retained.
The original 24-source primary remains not estimable.

Labels are available future coordinates for fitting and evaluation only.
Partially labelled trajectories are scored with the existing available-label
ADE, not silently completed 12-step paths. Unknown futures contribute the
original conservative moment bounds but no invented event label. No central
velocity, future endpoint feature, test-built goal or test normalization is
introduced. Previous causal construction is cached_verified; new fitting
source scopes and inference replays were checked again.

The data are exposed SOURCE development. No new H readout, new official role
assignment, independent final confirmation or reserved-data access occurred.
Native annotation-order image pixels are not verified seconds, meters or
human-gold annotations. Source groups are not verified IID physical scenes.

## Intended Use And Limits

Use these checkpoints to reproduce the negative cost-learning comparison,
not to replace a deployed policy. Inference permits only declared causal
packet keys. The full original risk/support gate fails; empty views remain
NA. Probability factors are not calibrated confidence guarantees. Pooled
harm ratios and bootstrap intervals are conditional development diagnostics.

The initial JSON receipt error and unchanged resume are preserved. All full
training, evaluation and exact replay completed in native arm64 Torch. There
was no NumPy substitute, reduced training budget or relabelled quick run.
Weights/arrays/third-party material remain private and ignored by Git.

The main paper PDF does not yet include this addendum. An independently
recreated anonymous raw-data cold start and a successful final method are
still absent. Deployment, submission readiness, Stage5C and SMC remain false.
