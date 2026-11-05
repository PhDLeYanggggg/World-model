# Preflight Fixture Amendment, Before Real Training

The strengthened gradient test initially failed: the inherited synthetic
easy target was zero for its first legal row, so both gradients were zero.
The earlier equality-only test was vacuous. The test now sets a known positive
easy budget-excess label before computing both gradients and requires a
nonzero original gradient as well as the exact10x ratio.

The numerical model, loss, fixed multiplier, data roles and evaluation rules
are unchanged. No real pilot, training or new held evaluation had begun.
The initial combined run was1failed/13passed, exit1. The initial standalone
three-test pass did not include this strengthened nonzero requirement.

The first registration.json is retained. The operative registration_v2.json
binds the corrected test and the runner's new registration filename. It is
created before any real fitting and must be committed before fitting. This
is a test-fixture correction, not a scientific outcome or parameter change.
