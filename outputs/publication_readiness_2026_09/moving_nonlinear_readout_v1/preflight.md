# Preflight Before Real Model Fitting

Native arm64 Python, sklearn 1.8.0 and joblib 1.5.3 are available locally.
The first scoped test run had two failures and eleven passes. Both failures
came from a synthetic fixture that omitted source_sites and fit_ids, which
the checkpoint binding correctly requires. The model code, scientific
configuration, source roles and thresholds were not changed to bypass them.

After completing those synthetic fixture fields, all thirteen scoped tests
pass in 4.00 seconds. They include real estimator fitting and exact interrupted
versus uninterrupted trees/predictions, a real toy neural encoder/readout,
unchanged future-target-independent inference and original decision ordering.
These tests do not establish real-data model improvement.

The first input-only registration preceded that fixture correction. Its bytes
are preserved as registration_preflight_before_fixture_fix.json. The effective
registration.json is regenerated with the corrected test binding before any
real-data pilot or held-source outcomes. No scientific result is overwritten.

Real training, held-source evaluation and deployment are not completed by
this preflight note. Reserved tests remain closed; original rules unchanged.
