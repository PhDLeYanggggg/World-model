# Source-Held Error Trial Gates

| Check | Result | Evidence |
|---|---|---|
| Original 2% rules and support retained | pass | registration_v2.json; no identity certificate |
| Real scratch teachers | pass | 8/8 at2,000updates; teacher_freeze.json |
| Own-source discovery normalization | pass | fit IDs and common/extra source exclusion checks |
| Discovery frozen before students | pass | teacher_freeze.json |
| Matched students | pass | 6/6 at2,000updates; same warm weights, sampling, data and optimizer |
| Actual encoder and optimizer updates | pass | result_summary.json; checkpoint audit |
| Checkpoint recovery and full refits | pass | pilot.json; teacher/student_full_refit.json |
| Three held readouts replay exactly | pass | verification.json |
| Prospective expansion screen | fail | only1/3contexts pass; aggregate contrast versus affine negative |
| Original full risk/support | fail | 0/17recording views for each new student |
| Easy preservation | fail | worst defined64.1321% for source-held errors |
| Original24-group primary | NA, not pass | missing reference support remains |
| Independent confirmation | not_run | H and reserved domains remain closed |
| Deployment / submission readiness | false | no safe improvement established |
| Stage5C / SMC | disabled | neither executed |

Engineering completion is separate from scientific success. Completion bounds
are not sampling CIs. The full historical test suite was not rerun;21focused
tests passed in2.39seconds for this change and its direct dependencies.
