# Recover The Original Fitting Roster

The initial prefix PID 15096 and train PID 15129 exited 1 at roster validation,
before model or optimizer creation. Fold 2 declares ten training groups, but
component 068 has zero past-eligible 8/12 windows in the immutable parent
population. The original model's actual row sampler had nine groups. The new
guard verifies the declared roster, the one original zero-support group,
actual normalizer groups and exact global fit IDs separately.

No row was removed or added; the empty group stays in the declared scope.
No update or outcome readout was performed in the two failed invocations.
Original objective, initialization, 300-update arms, draw stream, source
exclusions, held rows and scoring rules are unchanged. This is a forward
registration repair before fitting, not a new result or fallback run.
