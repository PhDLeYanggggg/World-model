# Verification Record

## Executed Experiment

The registered run fitted eight Torch risk networks and eight ridge controls.
There were 5,792 primary optimizer updates and 32 visits to every fitting row.
All 48,061 known proposals were evaluated. No new trajectory model, reserved
test readout, full-policy certification, Stage5C or SMC was run.

| Phase | PID | Seconds | Peak RSS bytes | Evidence |
|---|---:|---:|---:|---|
| Real resume pilot | 48213 | 1.221318 | 680198144 | Exit 0 observed; interrupted and continuous two-epoch states agree |
| Primary training | 48265 | 10.272005 | 1285865472 | Exit 0 observed; all eight neural and eight ridge fits complete |
| Full refit verification | 48345 | 10.442162 | 1249574912 | Exit 0 observed; all models and predictions reproduce exactly |
| Read-only replay | 48759 | 2.342508 | 1012236288 | Completion receipt verified; shell exit code not captured |

Replay forbids fitting, reconstructs predictions from saved models and fixed
references, and verifies all 22 then-existing public files remain unchanged.
That count precedes this verification note and subsequent prose-formatting
edits. No rerun or hidden optimization is represented by those documentation
changes. The replay PID is no longer live. Its private receipt is
`data/stage_cvpr2027_experiments/moving_neural_transport_v1/replay_execution.json`.

## Checks

Nineteen focused tests passed in 2.38s: neural fitting/resume, transport,
reporting, match attribution, track references/reporting and boundary audit.
This is not a rerun of the entire historical test suite. The final reporting
edits change prose only, not trained code, predictions or decision rules.

Verified result SHA-256:
`c7991cf9d80f5b99d06a1a083be90198ab13c7f49fc11cde84eb191500b50df0`.

The compiled main paper remains unchanged, SHA-256:
`d9babe3ddbac858bcc1ac30e3ba98b332bed625c78ff9231d1d42a3e30b11772`.
The separate addendum records negative development evidence, not a new main result.

## Scientific Status

The primary information screen passes 0/3 contexts. Refit and replay establish
computational reproducibility, not independent confirmation or efficacy.
All/easy harm, easy-net and support requirements stay unchanged; empty support
remains NA. No identity-fallback exemption is introduced. Deployment is unchanged,
the original 24-group primary remains NA, and submission readiness remains false.
