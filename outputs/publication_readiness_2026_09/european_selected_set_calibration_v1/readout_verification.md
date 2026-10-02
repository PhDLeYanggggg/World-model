# Independent Readout Verification

This additive verifier does not modify the frozen calibration runner, policies,
thresholds, coefficients, source roles or primary comparisons. No new fit and no
additional scheduled job. It may read results only after job37714473 is COMPLETED
with exit0:0,72groups and exact scheduled replay.

Read the72 hash-bound input packets and output groups from owned CREATE storage
into bounded RAM (128MiB binary-input/output cap); no local array, model or fold
cache. The original10GiB local numeric-cache reserve remains. Only the lightweight
summary (under64KiB), joint-OOF scalar details (under256KiB) and verification
receipt are saved locally. These are light aggregate metrics, not row caches. A failed access
or missing completion is not a scientific failure and never triggers resubmission.

The reader independently reconstructs causal eligibility from the frozen margins
without calling the production inference function. Original source OOF coefficients
are read from their hash-verified old calibration files; new OOF coefficients are
from the completed outputs. Every held recording must be absent from its fitting
recordings. Eligibility has no labels or future-availability argument. Both sets
must reproduce exact action hashes and new actions must remain parent subsets.

Risk masses are recalculated using scalar `math.fsum`, independently of the
production aggregation function. The original2% selected-reference denominator
is retained. Check all completion-bound fields, reasons, known/unknown counts,
nonempty coverage and utility. Floating aggregate agreement uses the existing
1e-10 absolute/relative tolerance; action hashes and discrete decisions remain
exact. Independently check summary counts and recompute its3000 locality-bootstrap
draws, averaging dependent views within locality before resampling.

Passing this verifier proves numerical consistency at this scope, not model
superiority, causal transport, independent confirmation or a physical-safety
guarantee. Empty/undefined selection is not a risk pass. New calibration still
requires its registered source-OOF utility/risk comparison; resubstitution cannot
rescue a negative held-recording result. Stage5C/SMC remain off.
