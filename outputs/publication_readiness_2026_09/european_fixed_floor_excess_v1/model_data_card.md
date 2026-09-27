# Fixed-Floor Signed-Excess Risk Head

## Material Passport

Fresh108 risk heads,216,000updates. Cached_verified108 matched MSE controls,
nine frozen forecasters, protected damping, fixed ridge utility and all fitted
normalizers. This is a development-only selector-risk experiment, not new
trajectory/JEPA training or a deployable world model.

## Inputs and Roles

380causal motion, interaction and candidate-rollout features; no future labels,
future validity masks/target latents, central velocity or test-derived goals
as inputs. Four forecaster-training, four floor-controller-training, two
risk-fitting and two readout sources are disjoint per fitted model. Only the
twelve previously opened source localities are used. Unknown-label rows are
excluded from supervised draws but retained in inference and query-count matching.

Observation8/prediction12 at raw-frame stride12. Image-local detector-silver
annotations. Partial and complete future-label support are separately reported.
Independent selection/calibration/confirmation roles remain closed.

## Architecture and Objective

380->64 GELU->4,24,644parameters. Softplus reference score bases and sigmoid
fractions of the causal rollout-disagreement envelope for harm bases match the
moment-MSE controls. The ONLY matched training change is mean squared loss on
all/easy `Hhat-0.02*Rhat` versus observed `H-0.02*R`. No extra moment anchor or
tail weighting. Loss averages two signed combinations versus four moments in
the control; these are different objectives with the registered learning rate,
not a proof that individual components become better calibrated.

Only signed combinations are supervised/identified by this loss. Individual
output components and their ratios cannot be interpreted as calibrated costs
or probabilities. The causal envelope bounds candidate-error difference, not
physical danger. Fixed movement/utility/input-support and all/easy2% predicted
budget screens preserve the protected floor when neural use is rejected.

## Optimization and Recovery

Fixed2,000updates,batch256,AdamW lr0.0003,weight_decay0.0001,clip_grad5. Native
arm64 CPU4/interop1,workers0. Atomic checkpoints every500updates preserve model,
optimizer,RNG,draw counts and loss monitor. Pilot100updates are inside the
budget. No held early stopping/checkpoint/hyperparameter selection.

Reused controls are matched by initial parameters, full sampled-row counts,
sampling/Torch RNG states, normalizers, settings and seed. Three forecaster
seeds and all six two/two rotations are retained; they are not216 independent
scenes or a fresh multi-dataset confirmation study.

## Evaluation Limits

The fixed12-source bootstrap uses3,000draws after averaging dependent views.
Unadjusted development intervals do not provide simultaneous or conditional
safety. Empty denominators remain undefined. Matching counts within a current
scene query is a diagnostic control, not a learned scene-joint allocator.

No full legacy-suite or cold raw-data rebuild is claimed. Models and private
caches stay local; only code/configs/aggregate reports are published. No metric,
seconds, human-gold, physical-safety, true3D or foundation claims. No deployment,
Stage5C execution or SMC. Read results/gates before interpreting any model gain.
