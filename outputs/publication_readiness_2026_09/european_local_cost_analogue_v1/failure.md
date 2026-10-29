# Preserved v1 Diagnostic-Writer Failure

Registration/code commit: c9d82416. Real pilot PID80933 exited 0 in18.213214s,
peak5,314,772,992 B. It verified7,572/7,572 support, unchanged frozen predictions,
neighbors and normalization; no scientific screen was evaluated in the pilot.

Full run PID81249 exited1 before writing its first group receipt:

    scripts/run_m3w_local_cost_analogue.py:65, diagnostic
    known = np.isfinite(y).all(1) & support
    NameError: name 'np' is not defined

No complete.json, summary or group output exists for v1. Local neighbors and
predictions were calculated, but no complete scientific result is claimed.
The process is terminal, not slow or awaiting access. This is an implementation
failure in diagnostic accounting, not a runtime deadlock or failed hypothesis.

A new runner regression test reproduces the identical NameError with known and
unknown queries. Existing33 tests passed because they covered the estimator,
receipt arithmetic and reporting but not this runner path. That coverage gap
is preserved and repaired, not hidden behind the passing tests.

The v2 amendment changes only the missing alias to api.np, adds the runner
regression and uses a new output namespace/registration. Numerical objectives,
features, neighbors, query roster, arms and scientific gates remain unchanged.
The original protocol, registration, pilot and code commit remain recoverable.
