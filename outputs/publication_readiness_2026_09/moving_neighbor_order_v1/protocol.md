# Neighbor-Slot Model Diagnostic

This own-B-only follow-up is registered after paired-gradient projection failed
its prospective training screen. It uses the three unchanged joint-control
models, not a model chosen retrospectively for utility. No new V/H readout.
Original 2% rules, support, labels, proposal masks and splits are unchanged.

Each agent retains the same eight neighbor histories and their masks, but their
storage slots are cyclically shifted by all offsets 0..7. Ego, floor, candidate,
action, query, coordinates and targets are unchanged. Use original own-B
normalizers and recompute causal kinematic features. The old architecture is
not claimed to be permutation invariant: nearest-neighbor ordering was a valid
input convention. This is a representation stress test, not proof of a bug or
real-world distribution shift. Slot-specific normalization may itself amplify
the perturbation; clipping is reported, not concealed.

Report all eight views and their fixed arithmetic mean of normalized risk and
decision scores. Do not choose a favorable shift. Measure own-B risk MSE,
observed cost-weighted joint decision errors and changes in admission. Averaging
is diagnostic inference only, not a new trained model or deployment upgrade.

A separate permutation-aware training hypothesis is supported only when at
least two of three contexts show >5% mean admission changes across the seven
nonidentity shifts AND at least two of three fixed-average outputs reduce MSE
without increasing joint cost. This is a fitting feasibility screen, not the
original safety criterion. No extra training is justified by sensitivity alone.

All nine previous fits and failed held results remain immutable. No unknown
target is fabricated or dropped from the risk regression calculation; unknown
outcomes are excluded only from observed decision-error metrics. Native
obs8/pred12 image coordinates, no metric/seconds/true3D/foundation/human-gold
claim. Original24-group primary remains NA. Stage5C and SMC remain disabled.
