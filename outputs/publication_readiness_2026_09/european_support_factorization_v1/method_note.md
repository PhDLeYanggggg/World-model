# Fixed-Support Factorization

This study isolates three costs within the previously rejected support rule. The registered runner and decision banks are immutable; this note explains them without adding a policy.

For each indexed agent at the current frame, the frozen stop controller chooses either neural forecast N or protected floor D. Its selected agents form S. Only these agents can remain selected under any new factor or matched control.

The three causal features are latest observed speed divided by mean observed speed, mean nonzero-pair heading change, and mean N-D future-rollout disagreement divided by twelve times mean past speed. The third feature uses model predictions, never a future observation. Existing per-fitting-source, per-motion-state boxes use the 1st and 99th percentiles and require at least 32 rows. No held-source statistic fits a box.

For an agent i, H(i) is the set of fitting sources whose first two box axes contain its features. D(i) is the corresponding set for the third axis. Each retained rule is intersected with S:

| Rule | Required support |
|---|---|
| History | At least two sources in H(i) |
| Disagreement | At least two sources in D(i) |
| Separate | Both conditions, potentially different source pairs |
| Joint | At least two sources in their intersection |

The joint rule exactly reproduces the preceding combined guard. The stop, joint and joint matched controls are frozen replay anchors. All factors have risk-ranked and fixed-random controls choosing exactly the same number of agents within each recording/current-frame query. A query with quota zero or its full eligible count cannot identify ordering quality, so flexible-quota counts are reported separately.

## Exact Error Accounting

For a removed selected prediction, let delta = error(N) - error(D). Positive delta is avoided harm; negative delta is lost benefit. Within each source locality and fixed evaluation subset, use the same total floor-error denominator for both. The difference of these two normalized sums equals the change in gain over D in percentage points. It is not a relative percentage gain over the stop controller.

Removals are partitioned into history-only failure, disagreement-only failure, both marginal failures and same-source-overlap failure. These four categories sum exactly to joint removals, including cases where both marginal supports pass but no common source pair does. Retained and outside-stop rows are separate categories. Unknown outcomes remain unknown; they can be counted as indexed rows but cannot contribute zero-valued errors.

## Evaluation Boundary

Two normalization modes, two target-parent families, three folds, three previously trained seeds and two event targets produce 36 correlated views per policy, not 36 independent experiments. Thirteen policies per parent yield 936 views. Each error comparison uses a fixed eight-locality roster with 3,000 paired locality-bootstrap resamples. The full study covers twelve previously opened source localities. Complete-history indexing, overlapping windows and partial future labels remain limitations.

History support and predictor disagreement can be disentangled algorithmically in this experiment. Their causal origins cannot: forecaster producers, fitting budgets and datasets remain unchanged. There is no threshold selection, new neural training, calibrated-risk claim or deployment promotion. Independent selection/calibration/confirmation remain closed. Image pixels and raw-frame 8/12 only; Stage5C/SMC remain off.
