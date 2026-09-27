# Report Serialization Repair

During fitting-summary inspection, before the held-source aggregate report,
standard JSON encoding rejected a NumPy int64. The count of views in which
L2 improves MSE but worsens log-mass error is accumulated from NumPy booleans.
The arithmetic and count value are correct; its representation is not a
standard JSON integer. Existing tests had not exercised this serialization.

The hash-bound registered report and runner stay unchanged. The new entrypoint
`scripts/report_m3w_european_cost_mass_native.py` converts only this named
diagnostic count in the returned fitting summary to a Python int, asserting
exact value equality. It then calls the original report writer. It changes
no fit, coefficient, frozen prediction, target, metric, bootstrap, contrast,
guard or gate. The temporary report-function adapter is restored on exit.

Regression tests reproduce the original JSON error, check exact numerical
and input-state preservation, require JSON round-trip equality and reject
noninteger counts rather than silently truncating. Use this entrypoint for
reporting and byte-replay. Independent roles remain closed.
