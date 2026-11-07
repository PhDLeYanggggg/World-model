# Causal Insertion With Optional Withdrawal

The count-preserving exchange recovered no recording, and its
label-aware capacity in missing-reference views was only 5/35
(8/35 allowing a later withdrawal). Keep the count-matched
result as a control. Test whether a learned insertion followed
by an optional withdrawal helps under the same risk contract.

Reuse the frozen forecasters, direct policy and exposed own-B
recordings, seven fit, six calibration and one source-held per
fold. At most one insertion, only if its current query has space
under the existing cap. After that, allow at most one withdrawal
of an original action in the same or a later query. A withdrawal
cannot make room for an otherwise illegal insertion. The original
output at a past query cannot be revised. No model sees later
queries, true reference availability or recording totals.

Fit two 32-wide neural heads, each 220 steps at batch 256. The
insertion head learns whether an action with observed easy
reference can produce a new original-rule pass, alone or with
one feasible withdrawal. The conditional head sees the selected
insertion's past-only features, the current withdrawal's
past-only features, and their query gap. It learns non-loss
repairs beyond what the insertion alone would have achieved.
For supported insertions sample one available positive and one
ordered negative withdrawal; add sparse negatives from insertions
without observed easy support. Use fitting-only labels and
cross-fitted upstream features. Reserve one quarter of each
mini-batch for positive examples if present. These model scores
are not calibrated probabilities.

Donor-only threshold search requires new passes, no lost passes,
no added signed easy violation and nonnegative paired lower
utility. Compare the original policy, independently calibrated
insertion-only, the optional policy's exact insertion without
withdrawal, and the optional policy. The last two are the primary
matched contrast. Apply the same insertion and withdrawal cutoffs
chronologically. Report per-query cap checks, passes, missing
reference views, changes, paired lower utility with a 3,000-draw
source bootstrap, and calibration availability. Do not count
fallback or empty effects as evidence for withdrawal.

Development only: 102 views repeat 34 recordings across three
forecast seeds and 14 source groups. Preserve the 2% and support
rules and retain failure. No independent outcomes, deployment,
metric, seconds-level, Stage5C or SMC claim follows.
