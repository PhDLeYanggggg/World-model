# Whole-Recording Supervision for Causal Forecast Insertion

The factorized support/value model improved its first-query local
probe but only picked one whole-record-feasible action in 35
missing-reference views. Test whether directly teaching one-action
recording rescue improves that choice. This is development on the
same exposed own-B recordings, not independent validation.

Reuse the frozen forecasters, direct decisions, legal one-insertion
roster and existing seven fitting, six calibration and one
source-held roles per fold. Cross-fit upstream features within
fitting sources. All heads share candidate sampling, source/recording
weights, architecture width (32), positive loss weight (8),
220 updates and batch size 256. The local model predicts two binary
targets on the same actions: observed easy reference with nonnegative
lower utility, and a new whole-recording pass with nonnegative lower
utility. A second model adds the query index and prefix sums of the
number and estimated utility/all/easy costs of earlier direct actions.
Neither model receives recording totals, future queries, or outcomes.

Compare ranking by local target, whole-record target on local
features, and whole-record target with the causal prefix. Apply
the same local-score guard, one-candidate-per-query budget and
calibration grid as the prior experiment; the learned scores are
not claimed to be calibrated probabilities. The original scorer
verifies the frozen choices. Select cutoffs on calibration sources
only, requiring new passes, no lost passes, no additional signed
easy violation and nonnegative paired lower utility. No deletion is
allowed in this matched one-insertion experiment.

Primary component contrast: first eligible query whole-record
non-loss rescue, whole-target vs local-target rank, paired
3,000-draw source bootstrap. Secondary: incremental prefix
effect, whole-record passes, recovery of missing support and
paired lower utility. Average three forecast seeds within sources;
102 repeated views are not independent samples. No changes to
the 2%/support contract, deployment, Stage5C or SMC.
