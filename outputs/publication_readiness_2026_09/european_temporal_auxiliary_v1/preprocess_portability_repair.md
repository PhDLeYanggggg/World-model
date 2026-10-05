# Measured TRAIN Scalar Portability Repair

CREATE diagnostic 37798223 completed 0:0 in 29 seconds. All 24 checksum-verified
TRAIN packets were recomputed; 15 have differences, exclusively in the scalar
float64 `scale`. The maximum absolute difference is 7.105427357601002e-15 and
maximum relative difference is 3.902099911572964e-16. All other preprocessing
fields, shapes and dtypes are exactly equal. No validation or independent rows
were read, no optimizer update occurred, and no checkpoint was created.

The scalar is a positive weighted reduction. Execution revision 6 permits at
most four float64 ULPs in its **recomputation check only**. This is not a general
allclose replacement: other fields remain exact, changed shapes/dtypes fail,
and zero, negative, nonfinite or non-float64 scalar values fail. Five-ULP changes
fail. This bounded portability rule is fixed before any new training result.

The original transported TRAIN preprocessing remains authoritative and is
returned unchanged after checking the recomputed value. Training normalization,
targets, initialization and loss therefore still use the original frozen
statistics, not an updated or rounded scale. The optimizer source and global
exact comparator are unchanged. A source-scoped context is restored even on
interruption; DataLoader workers remain zero. Synthetic comparisons cover all
three arms and interrupted/resumed fitting and require exact model, optimizer,
sampler, trace and preprocessing equality (except elapsed time).

The failed pilot 37797054 stopped at this comparison before optimizer creation.
Its script, submission records, initial heartbeat and events are archived, not
discarded. A guarded repair requires the exact terminal failure, successful
diagnostic accounting, matching diagnostic hash, zero checkpoints, no progress
events and no active M3W job. Only the portable entry point, new scalar guard
and their two manifest bindings change; source TRAIN arrays and the original
numerical training implementation do not change.

After committing the new port and reader registration, run the guarded repair
then explicitly submit the same pilot. Full training still requires its actual
losses, exact resume and measured resource gates. This portability diagnosis
and synthetic equivalence are not neural improvement or completed training.
Independent roles remain closed; Stage5C and SMC remain off.
