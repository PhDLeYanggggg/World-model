# Pending Evaluation Decision

Status: proposed for explicit user decision, not adopted. No new primary score
or favorable model outcome has been computed under this proposal.

## Reason

The old transform uses max(history path length, extrapolated last-step motion,
0.001) as the per-query denominator. Static-start windows consequently dominate
the normalized aggregate, despite contributing less than 1% of native-pixel CV
error in the broader explored SDD source population. This is not a future leak;
it is a consequential weighting and task-definition choice.

## Recommended New Registration

1. Keep the approved offline obs8/pred12 task, membership, data roles and split
   boundaries. Do not remove static, difficult or unsupported-label queries.
2. Report deterministic K=1 ADE and FDE in each dataset's verified native units
   separately. Label SDD annotation pixels and external dataset-local coordinates
   honestly; do not pool unlike raw units or convert to meters/seconds without
   independent calibration evidence.
3. For cross-scene summaries, define the gain in each scene as
   100 * (1 - mean_model_ADE / mean_reference_ADE), then average equally across
   prespecified physical scenes. Within each scene use the same supported rows,
   masks and prediction budget for both methods. Report each scene's ADE/FDE
   alongside the dimensionless summary and its worst-scene result.
4. If a scene has exactly zero reference error, percentage gain is undefined.
   Report native absolute harm explicitly; do not add an arbitrary epsilon,
   silently drop it, or declare an easy-preservation pass. The aggregate
   percentage requires all prespecified denominators to be positive; otherwise
   label it undefined and present the per-scene outcomes.
5. Preserve old past-normalized ADE/FDE, including all unfavorable results, as
   secondary diagnostics. Keep no-intervention CV and train-selected strong
   baselines separate. Never select the reference from held outcomes.
6. Separate numerical input conditioning from scientific error weighting.
   Candidate training-scale parameters must come from training inputs only.
   Preregister the loss and any easy/tail safeguards before new fitting. Do not
   silently reuse labels/thresholds defined under a different target scale.
7. Use scene/recording-level uncertainty, three seeds and independent
   confirmation if acquired. Existing exposed scenes cannot regain independence
   through a metric change. Keep one development review cycle distinct from a
   final locked confirmation readout.

This proposal is an evaluation repair, not a claim that it will yield positive
results. If accepted, first implement tested paired metrics and document the
registration amendment, then rerun fixed causal controls and matched existing
predictors before selecting a new model family. If rejected, retain the current
primary task and focus explicitly on static-start probability/direction support.
No new threshold selection or metric-dependent training before this decision.
