# Failure Taxonomy After Raw-row Linkage

| Proposed cause | Observation | Conclusion |
|---|---|---|
| Cached future labels misaligned with tracker/frame |All318,969 row masks/coordinates match the raw recording records|No detected packing mismatch; do not regenerate the dataset to fix an unobserved bug|
| Too few future labels explains all harm |8,285 harmful occurrences have all12 labels; most harm mass is in this stratum|Incomplete supervision is not the whole mechanism|
| Harm is always one unstable time point |Only13.35% of complete-label harmful rows flip/tie after deleting one point|Most complete-label harm is not eliminated by that diagnostic; this is not proof of clean labels|
| Error preference changes over the horizon |40.19% of complete-label harmful rows have opposing early/late signed error|Temporal cost structure merits a separate hypothesis; no metric/target change yet|
| Poor past observations identify risk |Higher line residual and slightly lower dense prefix support within query; other intervals overlap zero|Past-quality predictive control justified, not demonstrated as a repair|
| Partial neighbors alone explain failure |Paired association interval overlaps zero; earlier matched real retraining had no supported ADE lift|Do not rerun the same neighbor-only repair|
| Future detector confidence/box variation proves label noise |Associations exist but no independent visual/identity adjudication|Cause unidentified; do not filter or correct labels from this association|
| No class changes means no ID switches |Class-change contrast iszero|Not valid: two people can share a class and tracker association can still be wrong|
| Pooled easy risk below2% means deployment safe |Pooling hides local failures and unknown outcomes|Original gates remain failed/unchanged; this is not a deployment readout|

## Next Experiment Boundary

Keep original whole-recording train/validation partitions and the current
source-only roles. Add only the fixed seven past-quality proxies, after matching
their causal row IDs. Fit on known train labels, do not filter rows from future
availability, and compare with the same cached parent. Evaluate cost quality and
matched-count conservative utility before transfer; retain unknown completion
bounds and the2% budget. No hyperparameter or feature selection from the future
quality columns. A new auxiliary needs its own frozen protocol and replay.

This is an exposed-development diagnosis. Eight localities support the paired
quality contrast and eleven support the harm-share statistic; the effective
sample is not the596,988 repeated occurrences. No causal noise attribution,
independent confirmation, new training or deployment improvement is claimed.

The smallest missing external evidence is an independently adjudicated track/
label-quality subset. None was obtained here. A new researcher-created review
without independent identity evidence must not be labeled human gold.
