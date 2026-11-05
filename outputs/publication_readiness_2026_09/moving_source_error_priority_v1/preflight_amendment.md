# Pre-training Cache Interface Repair

The first real pilot, PID80282, exited1 before optimizer initialization:
`ValueError: Fixed role-specific row population required`.
The original forecast-cache interface accepts only its complete registered
role population; a reduced source-role request cannot reuse that cache ID.
No discovery/student checkpoint or new V evaluation was produced.

The correction reads the original immutable full-B forecast bank, then takes
whole-source, whole-query subsets for inner training. Normalization and target
RMS are still newly fitted on B minus the omitted source; full-B weights and
statistics remain prohibited for discovery teachers. Row IDs are preserved.
A regression test rejects splitting a query. No metric, source role, loss,
model size, training budget, threshold or expansion criterion changes.

The initial registration remains. `registration_v2.json` is operative and is
committed before restarted real training. This records an interface failure,
not a failed scientific experiment and not a successful runtime pilot.
