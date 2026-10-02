# Frozen Forest Decoder Control

## Hypothesis

The current forest's decoder rescales benefit and harm proportionally when their
sum exceeds the row's causal disagreement envelope. The operation can lower harm
without lowering reference error, thereby admitting switches that fail the risk
criterion before projection. Test whether this mechanical possibility explains
false-safe source predictions. This is a falsification, not a confirmed defect.

## Single Factor and Fixed Population

Reuse all 72 frozen forest heads (upstream seed43; head seeds17/29/43), their
original training/validation partition and existing 380 causal features. Recover
unprojected five moments from the fitted trees and verify the original projected
predictions, IDs, target and decision hashes against committed parent receipts.
Only source-held validation rows are evaluated. No independent selection,
calibration or confirmation role is opened; no directional transfer is run.

Original: scale B,H jointly to B+H<=envelope, then clip B<=R, ER<=R, EH<=H.
Control: H'=min(H,envelope), B'=min(B,envelope-H',R), ER'=min(ER,R),
EH'=min(EH,H'). R remains unchanged. This keeps at least the original harm and
no more benefit. Its eligible set must be a subset of the original. It is not a
calibrated upper bound and does not guarantee safe intervention.

No fitting, neural training, threshold change, new margin or winner selection.
Do not reuse old residual calibration on changed forecasts. Original 2% selected
positive-harm/reference criteria remain, including the distinct easy-event risk.
Unknown labels stay unknown, with negative-envelope utility completion and
positive-envelope harm completion; inference never sees outcome availability.
Empty/undefined selections are failures, not passes.

## Readout Frozen Before Data

Count envelope-binding rows, original harm/easy-harm reduction, changed actions,
removed known harms and unknown selections. Report before/after observed masses,
completion risk and full support. Compare conservative utility at unchanged
coverage and against expected uniform thinning within each recording+frame query
to exactly the control's count. The latter is a linear expectation, not a
realized random policy or expectation of its risk ratio. Normalize contrasts by
the full known reference mass. This is not an ADE/FDE performance claim.

Average heads/controllers within locality, then 3000 paired locality bootstrap
resamples (seed43). All intervals are nominal exploratory summaries on 12
previously exposed development localities. Recompute numerical predictions and
readout twice. Independently check scalar completion arithmetic. No promotion on
this source diagnostic alone, even if positive.

## Prior Negative Controls

Older neural L2/mean-mass readout controls already showed that projection can
reduce rare-harm mass and that scalar mean restoration is insufficient. This
experiment tests the current forest's specific joint B/H feasibility operation,
not a new scalar multiplier, percentile-support filter, extra seed search,
iterative calibration round or claim that projection has not been studied.

## Resources and Boundaries

Native local arm64, four Torch threads, no loader workers or resource probing.
Pilot one fixed first group, then all72 with resume and per-group heartbeats.
No numerical arrays/checkpoints are written: existing local disk is below the
unchanged10GiB cache reserve. Only <=8MiB aggregate reports are permitted.
Preserve unrelated staged work and remote jobs. 12h runtime cap, no slow-run
scope downgrade. Cached assets are hash verified; contrast is fresh computation.

Protocol: obs8/pred12, raw stride12, image-local detector-silver. No metric,
seconds, human-gold, physical-safety, true3D, foundation or submission-ready claim.
Stage5C and SMC remain off. Independent confirmatory roles remain closed.
