# Model And Data Card

## Intended Use

Development research on whether a neural forecast should replace a fixed causal
motion baseline. This controller is not approved for deployment, motion control,
physical safety, clinical use or autonomous decisions affecting real people.
The primary and original recording-risk gates fail.

## Models

18real Torch SOURCE-trained cost critics,29,253parameters each: three actual
forecasting/training seeds17/29/43, three producer folds and row/context arms.
Predict five coherent reference-relative moments for six physical and one
frozen history-Transformer candidate. The context arm adds parameter-free
other-target pooling within a current recording/frame. Inputs contain causal
geometry/detector history and forecast diagnostics. No new trajectory dynamics,
JEPA pretraining, generative rollout or calibrated confidence is claimed.

The unchanged 2% all/easy predicted guards and existing budget-pooling solver
produce independent, uniform, matched-joint and free-joint decisions. A failed
solver/postcheck retains the feasible parent. The physical-action ablation
removes an action, not its features or retrains a model. Actual risk is evaluated
after freezing actions; a feasible predicted constraint is not realized safety.

## Data And Roles

EuropeanSquares local-image detector-silver trajectories,obs8/pred12 at12raw
frames per step. Source training uses the original149,510-row controller
whitelist/74recordings/12localities. Each cost model receives only eight sites
not used to train its own forecast producer. Feature normalization is SOURCE-only
and physical-candidate-only; loss bias/RMS use SOURCE fitting labels only.

Model selection retains38,102targets/28recordings/six localities. Those outcomes
were already exposed by the preceding forecast readout, not fitted by this
controller and not an untouched final test.37,254rows have some future labels;
21,434complete12,15,820partial and848wholly unknown. Available endpoints27,694.
Missing rows remain in causal decisions and completion diagnostics. Risk
calibration, final confirmation and DroneCrowd remain unopened.

No future endpoint, future-support mask, central velocity, future goal, test
normalizer or test-endpoint goal is an input. Coordinates/timing lack verified
metric calibration/effective seconds. This is not a metric or human-gold dataset.

## Results And Limitations

Matched joint known gain versus floor0.1733%, primary joint-over-independent
paired lower+0.0016pp withCIcrossingzero;21/252record-risk views supported,
worst known easy degradation7.02%. Broader world-model capability and safe
joint-interaction contribution are not established. Action counts show nonzero
intervention but not reliable safety. Per-agent-type generalization and actual
collision/composition guarantees are not evaluated here.

## Compute And Reproducibility

Nativearm64CPU4/workers0; full18fits plus two complete exact refits take541.08s
with14.85GBpeakRSS.141,312primary+14,336refit updates;everyquery>=4visits.
Original numerical-pilot failure and missing-directory inference failure remain
recorded; recovery never changes model/budget/risk or relaxes exact equality.
All18cost outputs and54frozen-action metric groups replay; no solver reexecution
claim. Public scalar rebuilding is not independent training. Full anonymous
from-scratch asset release is still incomplete. Private raw data, arrays and
checkpoints are not committed. Stage5C/SMC remain disabled.
