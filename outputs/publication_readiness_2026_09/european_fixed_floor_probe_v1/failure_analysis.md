# Failure Taxonomy

1. **Default-action mismatch: confirmed.** Frozen neural decisions have positive
   incremental benefit once their fallback retains protected damping. Do not
   compare a CV-default neural policy with a damping-default control and attribute
   the entire difference to neural forecasts.
2. **Reference target alone: rejected in this matched probe.** Floor-target
   screening loses0.01686% relative ADE versus CV-target screening. This persists
   with the same fixed floor producer on probe fitting and held rows.
3. **Selected moment calibration: failed.** The nominal2% screen selects a
   subset with4.910% positive harm/reference. Good net ADE and easy preservation
   do not cancel observed individual harm. All twelve source-average ratios
   exceed the budget. See the conditional-moment diagnosis for the frozen-score
   accounting; it is post-readout, not an additional confirmatory hypothesis.
   In the floor-target screen,65.36% of selected harm predictions are zero after
   clipping;23.89% of those selected rows actually incur harm. Reference cost is
   overestimated2.03-fold on average. Thus both numerator and denominator can
   make the decision falsely optimistic.
4. **Weak ordering: supported for this linear model.** Floor-gain AUROC0.535 is
   weak. Benefit/harm squared-error skill does not establish a stable lift over
   fitting constants, and easy-harm skill is negative. This is not proof that
   the full feature representation has no information.
5. **Relaxing intervention: unsafe.** Positive-only selection gives large
   average gains but damages easy cases and one locality on all-ADE. Raw neural
   forecasts also harm zero-CV rows. Threshold release is not a repair.
6. **Support/coverage: unresolved.** The screened probe uses only5.63% of rows,
   captures0.705 floor-normalized benefit points and misses20.541 oracle points.
   A99% diagonal support filter is heuristic, not reliable conditional coverage.
7. **Evidence limitation: unchanged.** Only12 opened source-training localities,
   detector silver and repeated overlapping windows. Unknown future labels are
   excluded from supervised fits, retained in inference and not given zero error.
   Independent selection/calibration/confirmation remains closed.

No threshold/model/deployment winner is chosen from this readout. Negative
ablation and worst-view easy errors stay public. Stage5C/SMC remain off; no
metric, seconds, physical safety, true3D, foundation or submission-ready claim.
