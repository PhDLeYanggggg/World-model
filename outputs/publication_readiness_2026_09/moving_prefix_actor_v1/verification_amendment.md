# Verification-Only Registration Amendment

Before any real-data pilot/training, one additional readout test was added:
retain unknown/partial rows and keep empty easy references NA. The second
registration attempt correctly rejected the changed test hash. Preserve that
initial registration.json; registration_v2.json binds the added test and this
note. The runner only changes registration wiring and checks the exact original
model/loss, config and protocol bytes. No input, target, checkpoint, budget,
statistic, risk rule or role changes. Zero model updates or held readouts occurred
before this amendment. Register and commit version2 before actual fitting.
