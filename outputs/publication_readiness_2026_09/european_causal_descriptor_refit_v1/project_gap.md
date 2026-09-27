# Research Goal Remains Active

This experiment completes a specific feature-augmentation test; it does not
complete M3W or the submission-readiness goal.

## Established This Round

- 108 real native-Torch risk-head fits, with matched controls and recoverable
  checkpoints, are complete.
- Added causal descriptors improve unmatched mean ADE and slightly improve
  held signed-score MSE, but worsen the same-intervention-count contrast.
- Easy net preservation survives; positive-harm control does not. The
  negative and undefined outcomes remain in the result record.

## Current Bottleneck

The current mean-error improvement is not enough. The intended method must
decide which neural forecasts warrant intervention and control harm over a
strong causal floor. Global score error, coverage and conditional allocation
quality are not interchangeable. The next diagnostic should separate missed
benefit from excess harm in exchanged same-query selections, not revisit raw
thresholds or add features without a falsifiable reason.

## Evidence Still Needed

1. A fitting-only, targeted gain/harm ordering repair that improves matched
   intervention quality while retaining the fixed easy and risk requirements.
2. A direct comparison of independent-agent, whole-scene and scene-joint
   decisions on the same forecasts, sampling and intervention/risk budgets,
   including interaction consistency and source-level uncertainty.
3. Strong compatible public methods, independently selected and calibrated
   policies, and untouched confirmation with the full producer chain excluded.
4. A final paper and reproducibility package whose principal claims follow
   from those experiments, not from historical exposed t+50 scores or gate
   counts. The present development material can support negative ablations.

Independent roles remain closed until a justified policy and selection rule
are frozen. Routine diagnostics and bounded repairs proceed under the user's
delegated authority; no new user audit is required for each operation.
Stage5C/SMC prohibitions and final author confirmation for actual submission
remain unchanged. No current deployment upgrade or CCF-A readiness is claimed.
