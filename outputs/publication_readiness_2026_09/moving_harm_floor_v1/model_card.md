# Harm-Weight Allocation Risk Head

## Intended Use

Controlled development research on whether a frozen neural forecast should
replace the motion floor. This model does not generate new trajectories,
train JEPA, or provide a calibrated probability or population safety bound.
No automatic deployment follows from fitting improvements.

## Fixed Model

Separate risk and decision encoders, width 64, 64,260 parameters, 492 causal
inputs. The prior latent columns stay zeroed. History, neighbor summaries,
frozen candidate rollout and current-query pooling are past/current only.
Future truth and observation validity are supervision/evaluation only.

Three own-B contexts, one producer seed, nine complete fixed-budget fits.
joint reproduces the previous joint loss exactly. uniform_harm strengthens
every positive label by the same factor. harm_floor strengthens only labels
below the weighted median positive magnitude. The last two allocate equal
total positive coefficient mass under the exact planned fitting query weights.
The median and multiplier use observed supported own-B rows only; no held
statistics, new scene goals, future inference inputs or central velocities.

Original risk MSE, unknown supervision and all safe/zero-target auxiliary
weights stay unchanged. The derived weighting is loss-only, not an inference
threshold. Both heads/channels must still satisfy the original deployed
decision and reference/support tests.

## Evidence Limits

The README and machine-readable receipts, not this architecture description,
establish actual completion. Same-data replay is not independent confirmation.
The existing original 24-group primary remains NA, not passed. Missing support
and empty controls remain NA; no identity-fallback certificate is added.

Native obs8/pred12 image-coordinate development evidence only, not metric,
seconds, true 3D, foundation or human gold. Stage5C and SMC are disabled.
Private weights and data stay outside Git; local permitted assets are required.
