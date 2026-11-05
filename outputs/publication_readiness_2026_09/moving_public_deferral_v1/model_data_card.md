# Model And Data Card

## Intended Use

A task-adapted public routing comparison in exposed-source research. Not a deployed trajectory generator, physical controller or certified safe policy. No existing deployable model is replaced.

## Model

54independently fitted width64 routing heads,27969parameters each, three producer seeds and original nested source partitions. The seven candidate trajectories are frozen. Local causal features, action code and same-query other-agent context produce one logit per action. The48shared-latent fields remain zero. The public loss comes from Mao et al. (ICML2024), not a new architecture claim. A separately fitted point critic supplies the unchanged guarded-policy risk test.

## Data And Roles

MovingDroneCrowd++ SOURCE only, observation8/prediction12 in native annotation order and image pixels. Original admission and source grouping are inherited with hash checks. Causal rows and sampling are matched to the parent point control. Each model and all learned preprocessing exclude its validation sources. This is exposed development evidence, not independent final test evidence. No reserved Moving, DUT or DroneCrowd confirmation data is opened. Future labels only supervise fitting and evaluation; eventual mask availability is not an inference input. Wholly unknown outcomes are excluded from this loss, unlike the parent bound-target loss.

## Safety And Limits

Required nonempty matched utility is not estimable, not passed. Original full-policy risk/support requirements fail.

Original2%positive harm/easy-net and support requirements are preserved. The original 24group primary remains not estimable, not passed. Missing reference support cannot be converted into measured safety by an improved average loss. No identity fallback certification is used. Softmax/logit scores are not calibrated safety probabilities. No true3D, foundation, metric, calibrated-time or human-gold claim is supported. Stage5C and SMC stay off.

## Reproduction

Configuration, registration, full training receipts and inference hashes are public; raw data, private arrays and checkpoints remain local. Resume requires the unchanged definition and checkpoint. Separate inference replay verifies108policy blocks and all prediction hashes. Use `.venv-pytorch/bin/python scripts/replay_m3w_moving_deferral.py` for a read-only repeat after original verification. Native arm64 CPU4/interop1/workers0; no NumPy training fallback or MPS-performance claim.
