# Execution and Evidence Scope

2026-09-21. This is a focused research-method review and synthetic mathematical
verification. It is not a model-training run, an independent evaluation or a
reproduction of the cited papers' experiments.

## Executed Locally

```sh
.venv-pytorch/bin/python scripts/verify_m3w_conditional_risk_identities.py \
  --output outputs/publication_readiness_2026_09/conditional_decision_v1/verification.json
.venv-pytorch/bin/python -m pytest \
  tests/test_m3w_conditional_risk_identities.py \
  tests/test_m3w_cost_fit_forensics.py
```

Both commands exit 0. The exact-rational verifier reproduces four constructed
examples, eight binary policies and four conditional-mean rules, and refuses a
zero reference denominator. Tests: 20 passed in 0.15s (12 new, 8 existing), no skips.
No Torch model initialization, training or forecast inference occurs. Python
3.11.1 was the existing local virtual environment. No CREATE job was required.
All required processes are terminal. The full historical report-writing suite
was not rerun for these isolated mathematical examples and documentation changes.

Implementation SHA256:
`434f2455f25dc01852d89be6b4315e6a292127d79e0d62a02aba106dc824b1a7`.

## Reused Evidence

The two earlier cost-fit analysis files were hashed again and match their
existing verification bindings:

- Transformer: `5d7e63098b5705e3c50e9c2e01aa3ac321edc607bfa7f84f24465e1b9429b024`.
- EqMotion: `2854a2fd50a55ccdd0af7ec2d879deedd3f6212b270f68963bbd536037b0e9a5`.

Their 12-head/24-eligibility summaries remain cached verified evidence. The
original raw-batch reconstruction and frozen-coupling evaluation were not rerun
or counted as fresh training. A first hashing command failed because the shell's
`C.UTF-8` locale is unavailable; retrying with `LC_ALL=C` completed. No source
artifact or result was modified by that environmental error.

The saved synthetic JSON replays exactly from the unchanged verifier. State JSON
parsing and no-training/protocol-pending flags pass. All 92 local Markdown links
in the new notes and working paper resolve; scoped Git whitespace checks pass.

## Fresh Literature Inspection

Only public primary-paper pages were accessed. No local model, data or private
report was uploaded. The review records exact source versions and inspected
sections; it is not a systematic review or human reading attestation. The Mao
publication PDF failed web parsing, so its publication metadata and author
preprint are distinguished. The referenced theories are not applied without
their assumptions. No experiment in those papers was reproduced this turn.

## Unchanged Boundaries

The metric amendment remains proposed, not adopted. No new data role, training
loss, threshold, model choice or test labels. No independent risk calibration,
confirmation, deployment or submission-readiness claim. Stage5C and SMC remain
disabled; no metric, seconds-level, true-3D or foundation claim is introduced.
