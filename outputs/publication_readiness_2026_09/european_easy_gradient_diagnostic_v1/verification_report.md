# Verification Scope

Fresh CREATE diagnostic job 37569222 and exact replay job 37569306 completed
with exit 0:0. All 108 group files match. Local checks confirm 432 identical
paired initial gradients, 5,184 scalar geometry identities, twelve fitting
localities and disjoint 4/4/2/2 source roles. Reports and SVG regenerate
byte-for-byte. Twenty-one scoped tests in four files pass, including a
finite-difference check, no parameter mutation, masked labels, role overlap
rejection and interrupted-training regressions from the unchanged parent API.

The seal binds 52 source/control files and seven public artifacts:
`e809260e0d8e01afd2dd1f37d1cfc46278671e5a21af483c07e546ffac2a18c7`.

Scheduler wall times are 3m19s and 2m56s. Slurm batch MaxRSS reports
6,756,628 KiB and 2,139,096 KiB respectively; the first Python process reports
2,147,220 KiB. These are different accounting scopes, not a single reconciled
memory estimate. The local collection requested before replay completion
returned a missing-receipt observation; no job was duplicated. Collection
was retried only after the same replay job was observed COMPLETED.

No new optimizer update, held-label read, threshold selection, independent
source access or deployment occurred. Full legacy integration and raw-data
rebuild were not run. This proves diagnostic reproducibility, not model
improvement, selected-risk control or submission readiness.
