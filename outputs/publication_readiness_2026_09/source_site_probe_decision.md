# Source-Internal Visual Information Diagnostic

## Decision Before Fitting

The prior matched image study completed without robust source-to-main transfer.
Its mixed window-level gains reversed under equal-agent weighting, and did not
beat the same training prior in both main sites. This follow-up asks a narrower
question: does the identical image representation add predictive information
when generalizing between physical sites inside the admitted source data?

This is internal cross-validation of previously exposed auxiliary training data,
not a new official split, validation set, calibration set or final test. The main
11,966-query forecast cohort, native8-to12 protocol and equal-site past-normalized
ADE are unchanged. No sealed role is accessed. Original SDD train40 only.

## Fixed Comparison

- Five held physical sites: bookstore, coupa, deathCircle, gates, hyang. All videos
  belonging to a held site stay out of that fit. Scoped agent IDs also disjoint.
- Same cached stationary complete-label cohort:22,374 queries,726 localIDs,
  36videos. 6,460 incomplete-label stationary queries remain unscored,notnegative.
- Same476-column past geometry,8pastRGB32x32 crops and coverage. Same CNN/MLP.
- Two paired arms: mask_only versus past_rgb;seeds17,29,43. Thirty fresh fits.
- Uniform training-window sampling with replacement;2,000updates,batch64,
  AdamWlr.0003/wd.0001,gradientclip5,atomiccheckpoint200. No hyperparameter or
  checkpoint selection. Fixed final budget;paired initialization and row draws.
- Normalization fitted only on complementary source sites. No main examples or
  labels used for fitting. Existing loader verifies their cached fit-role assets
  only; no new main evaluation. No identifiers enter model inputs.
- Label remains any future annotation-coordinate change over12stride12 steps,
  not human motion-intention gold. SDD +144rawframes is not time-equated to main.

## Analysis Fixed in Advance

Primary diagnostic contrast: equal physical-site mean of the row-mean Brier
reduction from adding RGB. Report all five sites and three seeds, never only
positive sites. Average seed losses, not ensemble probabilities. Secondary:
equal-agent loss, training-prior control, AUROC/AUPRC/ECE/logloss, training loss,
and probability-mean versus varying-probability Brier decomposition.

Use2,000paired physical-site bootstrap resamples for the aggregate and paired
video-block resamples within each site. Only five physical sites; overlapping
training folds and previously exposed fit data make these conditional stability
descriptions, not new independent confirmation. Report equal-agent sensitivity
and single-agent omissions separately, not as replacement primary estimates.

If RGB cannot improve inside source, prioritize supervision/visibility/learning
diagnosis before domain alignment. If internal gain is reliable but transfer
fails, investigate domain-specific appearance and label differences. A positive
classification result alone never establishes forecast improvement, intervention
safety, world dynamics or deployment. No Stage5C or SMC.

## Execution and Provenance

Inputs cached_verified using original array/schema/role hashes;support audit,
fits and analyses fresh_run. A100-update training-only pilot estimates local cost
and is resumed within the first fixed fit. Nativearm64CPU4/inter-op1/workers0.
Checkpoints,RNG,drawcounts,PID,heartbeat and resume receipts stay local. Exact
prediction replay and completed-resume zero-update checks required. Preserve
the unrelated staged Git worktree. No data,images or checkpoints committed.
