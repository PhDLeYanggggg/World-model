# Reference-Complete Agent-Track Forecaster

## Intended Use

Research-only parameterization ablation on previously opened European SOURCE
data. It is not a calibrated safety policy or a deployed correction specialist.
Existing deployment remains unchanged.

## Architecture and Inputs

The inherited grouped agent-track Transformer has 88,514 parameters. It predicts
twelve relative positions from eight observed ego steps, up to eight masked
neighbor histories, a train-selected baseline rollout and requested times.
The 476-column geometry store is converted to the fixed causal tensor schema.
No new image, goal, SDF, JEPA or occupancy head is trained in this comparison.

The only model change is a parameter-free output-radius rule covering causal
CV and damping .97 with a fixed 5% interior margin. The center and zero-correction
initialization remain the train-selected baseline. All corrections stay within
the declared past-derived bound. The radius does not consult future labels.
Coverage does not guarantee that training chooses a reference or controls harm.

## Training and Reproducibility

Nine fixed models, three source folds and seeds 17/29/43. Same 4,000 AdamW updates,
source-balanced draws, loss 2+2 and normalizers as the harm-aware control.
36,000 primary updates complete in 537.401863 s, peak RSS 1.580 GB. Native arm64
PyTorch, CPU 4 / interop 1, workers 0, atomic checkpoints and heartbeat. No NumPy
fallback is counted as neural training.

The affected-fold pilot/resume/coverage-off checks and full 4,000-step neutral
replay are exact. Six unchanged-radius models reproduce their original
scientific states and predictions exactly. All models retain matched sample
and RNG histories. All endpoints precede scoring.

## Results and Limits

ADE gain over CV: 7.595475%, versus 7.531916% for the harm-aware comparator and
8.461640% for the original ADE-only model. Easy degradation: 8.345904%, not safe.
The small paired gain is conditional source-development evidence, not an
independent result. Positive-harm constraints remain separate from net error.

The full action-bank probe is label-aware and does not turn this forecaster
into a successful selector. Its 148/148 physical-bank feasibility does not
upgrade any learned model. No new candidate-dependent cost head or independent
calibration is fitted. Neural and physical-only learned controls remain needed.

Released detector tracks and incomplete labels limit interpretation. Repeated
windows, contexts and seeds do not increase the twelve locality clusters used
for the conditional bootstrap. No human-gold, metric, seconds, true 3D,
foundation, world-model success or submission-ready claim follows.

## Artifacts

Code, configuration, protocols, hashes, reports and light metrics are versioned.
Checkpoints, predictions, geometry and verified TRAIN byte caches remain local
and ignored by Git. Reproduction needs the audited assets and original control
weights; this is not a public standalone weight release. See
[operating guide](reproduction_zh.md). Stage5C and SMC remain off.
