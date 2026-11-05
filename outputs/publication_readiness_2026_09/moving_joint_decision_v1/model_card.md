# Joint-Decision Risk Head

## Intended Use

An exposed-development experiment in deciding when a frozen neural forecast
may replace the existing motion floor. This head does not produce new future
trajectories and does not retrain JEPA. It is not a deployed safety certificate.

## Architecture And Data

Separate risk and decision encoders, width64,64,260 parameters. Each action
receives492 causal features: original428-vector context plus64 motion-basis
features. The previous latent columns are zeroed as in the parent. History,
neighbor summaries and frozen candidate rollouts use the current/past query
only. Current-query pooling is not future access. Own-B preprocessing and
the frozen source-excluded proposal masks are bound by hashes.

Three own-B contexts, one producer seed. A supported row supervises one frozen
proposal, not all legal actions. Futures and their validity are loss/evaluation
only. Observed labels retain their available-outcome meaning; no full future
is imputed. Unknown labels retain the original conservative supervision.

## Fixed Intervention

Both arms retain original risk MSE and decision objectives. The per_channel
control exactly uses the preceding composed loss. The joint arm changes only
the added observed-label term to the sign cost of the maximum over both heads
and both channels. The two derived target magnitudes are averaged across
channels, retaining comparable scale. The coefficient stays0.5.

An all-safe target with one exactly zero channel can now train that blocking
output. No zero-channel exception exists at inference. Both heads/channels
must still pass the original threshold and reference/support checks. Scores
are not calibrated probabilities or certified upper bounds.

## Training And Reproduction

AdamW lr.0001, decay.0001, gradient clip5, complete-query batch72, fixed2,000
updates per model. Same warm starts and query sequence across matched arms.
No checkpoint selection or threshold search on held results. All six fits
freeze before conditional development evaluation. Native arm64 CPU4,
interop1, workers0, checkpoint and heartbeat every100 completed updates.

Use the [operation guide](operation_zh.md) and machine-readable receipts.
The README, result_summary.json and gates.md report completion and outcomes;
the architecture description alone is not evidence of completed training.
Both actual100-update resume pilots have already matched exactly.

## Restrictions

Original2% all/easy positive harm, easy-net and support rules unchanged.
No identity certification; empty controls remain NA. Original24-group primary
not estimable, not passed. Reserved20 Moving/DUT/DroneCrowd/H remain closed.
Native obs8/pred12 image-coordinate evidence only, not metric, seconds, true3D,
foundation, or human gold. Stage5C and SMC stay disabled.

Repeated exposed-development improvements cannot establish independent
generalization. Lower fitting cost cannot establish safe deployment. Public
code/config/aggregate receipts exclude private arrays and weights; local
permitted data assets are required for replay. No model promotion is automatic.
