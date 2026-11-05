# Validation Receipt

Observed terminal exits were0 for pilot92728, complete training92869,
not_run-recording evaluate93829, fitting replay93881, attribution93946,
and attribution replay94001. Report generation and its unchanged-version
replay also exited0. No relevant experiment or test process remained after
these checks. This is completion evidence, not a scientific safety pass.

Twenty-seven focused tests passed in2.74seconds:
`test_m3w_full_objective_risk.py`, `test_m3w_full_objective_reporting.py`,
`test_m3w_kinematic_risk.py`, `test_m3w_annealed_risk.py`,
`test_m3w_all_easy_gradient.py`, `test_m3w_admission_priority.py`,
`test_m3w_signed_decision.py`. The full historical suite was not rerun.

Tests cover complete-objective gradients, actual resume, interruption without
serializing transient line-search weights, future-input rejection, failed
fitting-screen prevention of held prediction, fixed original rules, choice
before target readout, and decomposition of known/unknown and newly wrong/
repaired errors. Real-data replays supplement these synthetic/static checks.

All23 existing experiment evidence files were byte-identical after the final
unchanged reporter replay. This receipt is the subsequently added24th file.
Structured comparison confirmed research_state mean-loss reductions,
all/easy decision-cost changes, actual updates, gradient passes, readout status
and training duration match result_summary.json exactly.

Analysis SHA256:
`006d40b14d8ad0311a1fb935316c123a2229dc86ba0927d9c15cce5d21962aeb`

Fitting-attribution SHA256:
`e055abcd6b63f48e180fa503576643ecc3fe66337d056a91a90784870d6de8dc`

Private artifacts occupy approximately38MiB; public experiment files are
approximately164KiB before this receipt. Raw data and checkpoints stay private.
The unrelated staged-work sentinel remains unchanged. No independent held
result, convergence proof, deployment or submission-readiness upgrade follows.
