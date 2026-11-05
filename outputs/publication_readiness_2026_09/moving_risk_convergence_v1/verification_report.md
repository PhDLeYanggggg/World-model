# Verification Record

## Completed Execution

| Phase | PID | Seconds | Peak RSS bytes | Result |
|---|---:|---:|---:|---|
| Real recovery pilot | 49656 | 1.396060 | 685637632 | Exit 0; interrupted 128+128 updates exactly match continuous 256 |
| Eight continuations | 49711 | 25.403020 | 747159552 | Exit 0; 32,871 additional updates, all complete epochs |
| Eight repeated continuations | 49805 | 20.501314 | 737656832 | Exit 0; every saved model/optimizer/random state and loss trace agrees |
| Final development evaluation | 49864 | 2.282181 | 883490816 | Exit 0; final checkpoints only, 48,061 known proposals |
| Read-only replay | 50182 | 2.246538 | 881655808 | Exit 0; fitting forbidden, predictions and metrics agree |

The primary update count excludes the repeated verification and the recovery pilot.
Old 32-epoch parents remain unchanged. Continuing a finished parent into a separate
registered experiment is distinct from resuming an interrupted continuation.

The replay verified all 26 then-existing public files stayed unchanged. This note
was added afterward. The completion receipt is local at
`data/stage_cvpr2027_experiments/moving_risk_convergence_v1/replay_execution.json`.

## Tests And Artifact Checks

Twenty-four focused tests passed in 3.22s. Coverage includes exact recovery,
unchanged parent states, rejection of partial parents, exclusion of held labels
and held features from fitting, transport alignment, reporting, source references
and boundary diagnostics. The historical full suite was not rerun. Later report
rendering, metadata reconciliation and figure generation executed successfully;
the fitting/source-held comparison figure was visually checked.

Source coverage is cached_verified: metadata file hashes, parent preprocessing
source lists and fitting row counts agree. It is not a newly collected dataset.

Verified analysis SHA-256:
`2aa7463a61c483e66c56b3b9e8a144880d01b872e1d55d994a71048dae047e6a`.

Unchanged compiled main paper SHA-256:
`d9babe3ddbac858bcc1ac30e3ba98b332bed625c78ff9231d1d42a3e30b11772`.

## Scientific Limits

Engineering checks do not override the primary screen, which passes 0/3.
There is no independent confirmation, formal multi-seed main result or new
trajectory success. Original risk/support rules and empty-control NA remain.
No identity exemption, deployment promotion, original 24-group primary pass,
submission-ready claim, Stage5C execution or SMC follows from these checks.
