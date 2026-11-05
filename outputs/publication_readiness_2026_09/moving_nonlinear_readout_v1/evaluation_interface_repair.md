# Evaluation Interface Repair After Training Freeze

All six model fits completed before the first evaluation attempt. That attempt
exited 1 under PID 73351: the generic source-isolation check expected a
top-level preprocessor, but the new readout stores it in its frozen parent.
The traceback and failure log hash are retained separately. No new held-source
metric reduction completed before this exception; fitting diagnostics had
been computed. This is an engineering failure, not a negative scientific result.

The repair verifies readout and frozen-parent source/row identity, then gives
the unchanged exclusion check that parent's actual preprocessor. No check is
removed; overlap of fitting rows, fitting sources or preprocessing sources still
raises an error. No estimator, feature, target, source role, threshold or
scientific evaluation definition changes. Valid trained models are not rerun.

The original registration remains immutable. A separate amendment binds its
hash, the training-freeze hash and old/new runner hashes. The loader permits
only this runner binding to differ; all other original registrations and
scientific fields must match exactly. The amendment is committed before the
evaluation retry. The failed private log is preserved; retry uses a new log.

Regression coverage explicitly rejects shared fitting rows, shared training
sources, shared preprocessing sources and inconsistent frozen-parent identity.
This repair is not evidence of model improvement. Original 2% rules, support,
empty controls, old failures and closed H remain unchanged.
