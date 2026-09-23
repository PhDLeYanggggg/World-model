# Frozen Source Refit Before External Calibration

Registered on 2026-09-23 before any new fitting or reserved-source predictions.
This experiment prepares fixed predictors for later external work. It is not a
new architecture search, a positive-result claim, or independent evaluation.

## Why Refit

Existing development experiments have twelve outer-site models per family, each
trained on a different three-site complement. Choosing one of those outer-site
models after seeing external data would create an avoidable selection freedom.
Instead, refit the already-tested Transformer and pinned deterministic EqMotion
K=1 on all four exposed SDD source sites, using three fixed seeds per family.
This yields six unambiguously defined source-only predictors for a later matched
intervention comparison. Existing outer/pair-excluded producers remain unchanged
and available for honest out-of-fold cost supervision.

Use all 175,756 past-eligible queries from the 33 admitted original-training
recordings in coupa, deathCircle, gates and hyang. This entire set is explicitly
development-exposed; it cannot become final confirmation. No held-site claim is
made for these new full-source fits, and training losses are not validation scores.
Do not train a downstream cost head using these predictors' in-sample costs as
if they were out-of-fold outcomes.

## Fixed Models, Budget and Sampling

Use the exact existing native-coordinate architectures, conditioning, loss and
optimizer settings. Transformer: width64, heads4, layers2, motion-bounded output.
EqMotion: the already-used pinned author core, deterministic head0, width64,
channels64 and four layers, without future-informed best-of-K selection.
No new third-party program or weights are acquired. This is an adapted K=1
control, not reproduction of the paper's best-of-20 benchmark.

Seeds are17,29,43. Each fit uses4000updates of64rows: sixfits,24000updates and
1536000draws. Both families get the same per-seed source-uniform batch stream.
Sample a source site uniformly, then any indexed query uniformly within it.
Retain unsupported supervision rows with zero masked loss, and apply the existing
indexed-to-supported correction to preserve equal supported-site loss weighting.
Native-coordinate loss normalizers are fitted on these source sites only.

The final fixed-budget checkpoint is the endpoint. No early stopping, validation
selection, pilot-score selection, family winner selection or threshold search.
The pilot runs the first100updates of one registered model and resumes the same
checkpoint; it is a runtime estimate, not a smaller final training budget.

## Input and Output Boundary

The fixed interface observes8 sampled annotation steps and predicts12 future
sampled steps. SDD's source stride is12 raw annotation frames. Inputs are past
ego motion, observed neighbor history and causal CV rollout; labels and their
availability masks are loss-only. No scene IDs, future endpoint, future goal,
central velocity or test-derived normalization enters the prediction interface.
This refit does not add images or make a new multimodal-contribution claim.

DUT is calibration-reserved; DroneCrowd is confirmation-reserved. The source
reservation hash is bound before fitting. Neither reserved raw/cache arrays nor
their forecasts or predictive labels are opened. This source-only refit does not
settle their source-use, historical-exposure, site independence, adapter, risk
support or one-shot evaluation requirements.

## Runtime, Recovery and Checks

Use the verified arm64 .venv-pytorch, CPU4threads, interop1, workers0. Reject
Rosetta before Torch import. No hardware resource probing or worker multiprocessing.
Local execution is reasonable if the actual100-update training pilot predicts
completion within the author's12-hour allowance; otherwise assess CREATE before
submitting any job. Slowness alone is not permission to reduce the registered budget.

Reuse atomic checkpoints containing model, optimizer, sampler/Torch RNG state,
draw counts, loss factors, input identity and step. Save every200updates and
heartbeat every50. SIGINT/SIGTERM leaves the last atomic checkpoint resumable.
One owned lock prevents overlapping runners. Checkpoint/config/source drift is
an error, not permission to silently restart or relabel a new run.

After all six endpoints complete, verify finite model weights/loss traces,
all4000updates, exact train IDs and loss factors, total draws, and identical
per-seed family sampling. Check a fixed source-only input prefix for finite
predictions and replay identity, without treating in-sample error as evidence of
generalization. A separate numerical test compares the full-source loss factors
with the existing complement implementation wherever the same source is fitted.

## Interpretation

Successful completion establishes six reproducible fitted predictors, not their
external accuracy or protected intervention value. No accuracy CI is computed
from training samples. No new calibration, confirmation or deployment occurs.
The empirical2%easy limit, strong-baseline requirement and negative development
results remain unchanged. Physical scene/source uncertainty and at least three
seeds remain required for later formal reporting. Coordinates and horizons stay
annotation-pixel/raw-frame; there is no verified meter, second, true3D, foundation,
physical-safety, Stage5C or SMC claim.
