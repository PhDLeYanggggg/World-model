# Harm-Aware Agent-Track Forecaster

## Intended Use

Research-only matched loss comparison on previously opened European SOURCE
development data. It is not an autonomous control system, a certified safety
policy, or a deployable upgrade. The complete deployed policy is unchanged.

## Model and Inputs

- Nine dimensionless grouped agent-track Transformer producers: three source
  folds and seeds 17, 29 and 43; 88,514 parameters per model.
- Eight observed steps and twelve requested future steps, raw-frame stride 12.
- Causal ego history, masked partial-neighbor history, training-selected
  baseline rollout and requested times; 476-column frozen geometry schema.
- Motion-bounded correction around the training-selected baseline. Output is
  a deterministic image-coordinate trajectory, not a latent generative rollout.
- Future coordinates, errors and easy/hard labels are loss/evaluation labels
  only. No future endpoint, central velocity or test-endpoint goal input.

## Training

4,000 AdamW updates per producer, source-balanced sampling and inherited
training-only scale factors. Native ADE plus coefficient-2 positive-harm
penalties against the selected baseline and causal CV. No checkpoint selection.
Three seeds and all endpoints are reported. Native arm64 PyTorch 2.12.0,
CPU 4 threads, interop 1, workers 0; atomic checkpoints and heartbeat retained.

Exact neutral and resume replays check implementation comparability. Full
current-family training completed in 675.416 seconds with 1.633 GB peak RSS.
No NumPy-only substitute for neural training was used.

## Evaluation and Limits

Equal-locality ADE gain over CV is 7.531916%, below the original producer's
8.461640%. Easy ADE degradation is 8.433059%, still above 2%. Positive harm
remains separate from net degradation. The original controls and new readout
replay exactly; this does not make exposed development data independent.

The source population includes missing future labels, overlapping windows and
released detector tracks. Unobserved outcomes remain unknown. Bootstrap units
are twelve localities; repeated contexts and seeds do not increase that count.
No human-gold, metric, seconds-level, true-3D, foundation or submission-ready
claim is supported. Existing candidate-dependent policy heads are stale for
changed forecasts. No new independent calibration was performed.

The model cannot represent CV within its bounded output set on some fold-2
rows. The remaining unsupported 082/51 view has a different unresolved cause.
See the failure and output-bound reports rather than treating either as a
universal explanation.

## Artifacts and Status

Code/configuration, endpoint hashes, reports and light metrics are public.
Raw data, geometry, predictions and checkpoints stay outside Git. Reproduction
requires the audited local assets; it is not a public weight-only inference
release. `reproduction_zh.md` lists training, resume and exact replay commands.
Deployment, Stage5C and SMC remain unchanged/off.
