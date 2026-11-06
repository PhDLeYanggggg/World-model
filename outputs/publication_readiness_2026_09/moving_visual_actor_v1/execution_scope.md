# Input Access And Outcome-Readout Scope

Preparation opens past metadata and the causal image-context banks only. The
reported preparation flag `target_arrays_opened:false` applies to that phase.

All later phases reuse the generic producer-A loader. It opens read-only
memory maps and verifies byte/array hashes for the complete packed training
bank, including supervised targets and future-validity arrays. Therefore this
runner does **not** implement a filesystem-level target-unopened inference
process. The protocol's freeze-before-reading-labels wording means freeze
before diagnostic outcome scoring, not that no target file bytes are opened
for integrity verification. This distinction is recorded before actor scoring.

Forecast inference forwards only the existing causal geometry and the declared
46 past-image context inputs. Targets, future-validity masks and future
endpoints are not passed to the model. Source-specific normalizers use fitting
past inputs only; training loss factors use fitting labels only. All twelve
final donor prediction banks must exist and freeze before diagnostic scores or
component selection are computed. No prediction choice is revised after that
readout. No new own-B, C/H or reserved cohort is admitted.

The full data bank belongs to exposed producer-A development. Source exclusion
holds within each fitted pipeline, but this is not independent confirmation.
The distinction does not restore a fresh-test role or change any original
2%all/easy, signed easy-net, reference/support/NA requirement.
