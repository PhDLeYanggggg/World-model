# Interpretation And Remaining Gap

The matched representation contrast is -0.00030587pp.
Its conditional lower bound is -0.00062506pp.
The motion arm's gain against CV is -0.00084041%,
with 0 of12 individually positive held fits.

This distinguishes measured input variation, reduced forecast harm and actual
predictive improvement. Engineering support and exact replay cannot replace
positive forecasting evidence. A tiny positive hard-slice result alone cannot
justify intervention if aggregate and static-target behavior are worse.

Possible limitations remain distinct:32px low-pass crops; annotation regions
rather than segmentation; neighboring-object/camera/occlusion confounding;
scarce independent onset events; source shift; representation aggregation;
and the tested conditional decoder/objective. This experiment cannot identify
one of these as the sole causal failure. It does not prove all pixels are useless.

The prior native96px ETH/Hotel/Zara experiment was negative. Do not simply
claim that higher resolution will repair SDD or repeat a threshold sweep on
near-zero candidates. A separately registered probability probe has now fitted
16 logistic models. The motion features slightly improve larger-excursion
ranking (AUROC difference +0.01241), but worsen Brier by0.001513 on average and
on every held site. Absolute motion AUROC is only0.407-0.556. General nonzero
change prediction does not improve consistently. This does not provide a safe
gate or prove useful trajectory direction. Full probability results and the
11-row numerical label-boundary disclosure are in
[the complementary report](../source_box_motion_probe_v1/conclusions.md).

Eighteen of24 trajectory fits improve training ADE slightly but none transfer;
the other six do not even improve training ADE. This is evidence of limited
source predictability/transfer for these readouts, not proof of unlearnability.

The low-pass32px crop's median annotated box is9.34 by11.86pixels, smaller than
the fixed15px flow aggregation window along both axes. Measurement support
therefore cannot establish resolved body motion. That scale comparison is a
post-hoc limitation, not proof it caused the failure. Before a higher-resolution
training run, inspect whether independently defined past events retain body
motion at native resolution, with matched regional/quality controls and no
held-label selection. That native SDD measurement comparison is not_run.

Main/external confirmation and scene-level risk calibration remain unestablished.
No new model is deployed. Baseline rejection is a fallback, not neural success.
