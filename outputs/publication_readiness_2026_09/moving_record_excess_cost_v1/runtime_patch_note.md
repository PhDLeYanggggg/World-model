# Pilot Logging Fix Before Primary Training

The first real pilot (PID65082) exited1 at its first trace write: dict construction
supplied `total` twice. The objective and optimizer step were finite. The saved
pilot checkpoint is step0; its one unsaved point-objective update is rerun on
explicit resume. No primary model or new Hprediction had started.

The repair merges trace values before recording base_total and total. It does
not alter the objective, features, targets, grouping, sampling, initial weights,
optimizer, checkpoint binding or budget. Original scientific registration is
retained, while runtime_registration.json binds the corrected runner/module/tests
and this note. Configuration, controls, jobs and forecasts must match the
original registration exactly. Checkpoint identity retains the original
scientific registration, permitting same-state recovery of the saved step0.

A trace regression test and actual two-update synthetic fits with checkpoint
resume were added. They are engineering tests, not real-data model evidence.
The full real pilot still must pass all six resume cells and three original
point-loop comparisons before primary training. This was a logging exception,
not an OpenMP/SHM hang. No scientific threshold, data role or conclusion changed.
