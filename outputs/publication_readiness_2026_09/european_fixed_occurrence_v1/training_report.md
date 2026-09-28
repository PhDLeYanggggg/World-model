# Fixed Occurrence: Paired Training

Fresh native-arm64 Torch training:108 groups,216 heads,432,000 new updates.
The full invocation took 810.16s, resuming the pilot group; this
does not include the earlier pilot's elapsed time. All checkpoints
hash-verify. Initial states and sampled queries match between paired arms.
Occurrence parameters and predicted probabilities remain exactly frozen in every
fixed arm. The first full pair was retrained with exact non-timing states; the
other107 pairs were not independently retrained.

Both arms start from the same uncapped checkpoint, split into identical
occurrence/cost branches and reset AdamW. Only occurrence trainability differs.
This is not a test of splitting versus the old shared architecture.

| Arm | Complete fitting marginal objective |
|---|---:|
| Trainable occurrence | 0.00237859938 |
| Fixed occurrence | 0.00250782941 |

Fixed occurrence improves the complete fitting objective in2/108 groups.
All occurrence, row/query, conditional reference/harm and signed-bias components
are reported in training_report.json, with undefined values retained. These are
equal-group averages of source/query-balanced quantities; 108 overlapping views
are not 108 independent samples.
The fitting-only sign-screen counts in training_report.json are diagnostics, not
deployment actions: they omit overall risk, eligibility and joint utility.
14,904 arithmetic checks verify the source/query-weighted accounting.
Fitting scores are not calibration or transport results; no model or threshold
is selected from them. Only12 already-opened development localities; repeated roles and the three forecast seeds are not independent samples. Intervals are nominal3,000-draw paired-locality intervals, not simultaneous or independent-confirmation intervals. Independent selection/calibration/confirmation remain closed. The original selected-risk primary remains incomplete; a full-floor denominator cannot replace it. Image-local detector silver; obs8/pred12 with raw-frame stride12. No metric, seconds-level, human-gold, physical-safety, true3D, foundation, calibration-certificate or submission-readiness claim. Stage5C and SMC remain disabled.
