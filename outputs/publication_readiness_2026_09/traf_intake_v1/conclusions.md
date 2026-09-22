# External Scene Intake: TRAF Is Not Yet Eligible

## Material Passport

Date: 2026-09-22. Fresh local raw-annotation audit, exact rerun, separate CSV
recount and window verification. No new training, forecasts, calibration,
confirmation, or scientific role assignment. The previous conditional-cost
experiment and its failed per-scene easy protection remain unchanged.

## Verified Local Findings

The OpenTraj checkout identifies commit
`97ec1d5e1b579f26febeaa9083705952deac338a`. The audit hashes the actual local
annotation files and documentation; a checkout identity alone is not a claim
that every local file matches an upstream release.

- Thirty annotation files form 22 filename families. Their number is not the
  number of independent physical sites; that mapping is absent.
- There are 55,526 input frame lines. Of those, 468 repeat an agent identity
  within a frame and are excluded from structural counts; their recordings are
  quarantined rather than silently cleaned or admitted.
- The remaining 55,058 frame lines contain 592,266 agent-box observations and
  2,713 recording-local track IDs. These are before whole-recording quarantine.
- Only 159 tracks explicitly use the documented `ped` prefix. They contain
  65,858 observations, 62,646 overlapping complete 8-to-12 views at stride 1,
  and 39,004 at stride 12 over all phases. These are availability diagnostics,
  not a selected time protocol, independent samples or newly eligible data.
- A further 1,638 tracks have IDs without a recoverable documented class.
  `man`, `human`, `chair`, `desk`, `object`, `sccoter` and `scoote` also appear.
  No automatic relabeling merges them into pedestrians or known traffic classes.
- Twenty-seven recordings require identity/type review. The other three pass
  only this limited structural screen, not the remaining admission requirements.
- No identical file or complete relative-frame/box-track duplicate was found.
  This does not rule out overlapping clips, changed IDs or transformed copies.

The coordinate conflict is substantial: 587,251 of 592,266 parsed boxes fail
the README's xyxy ordering. All have positive third/fourth values, consistent
with a possible width/height interpretation but not proof of that convention.
The code explicitly refuses center/footpoint conversion. Some negative image
fields may be out-of-frame boxes; they are counted, not silently clipped.
The local dataset README states 20 FPS and the toolkit table 10 FPS. Neither
value has been established for these particular files.

The original paper distinguishes moving/static and front/top views; its reported
evaluation includes car-mounted frontal views. A traffic annotation file cannot
automatically stand in for an independently verified static top-down scene.
The paper's reported metric results do not calibrate these local boxes.
[TraPHic, sections 5.1 and 5.4](https://arxiv.org/html/1812.04767v4).

## Implemented Repair

The new parser preserves opaque IDs and raw box fields. It validates record
length, frame order, finite fields and duplicate IDs; reports undocumented types;
counts gaps without interpolation; and separates history-only support from full
future-label availability. An initial development parser assumed every ID used
the documented class prefix. Actual files also use numeric/opaque IDs, so this
was corrected before the final audit: unknown type is now explicit rather than
mistaken for an unreadable identity.

No targets or geometry are released for inference. No roles are assigned. The
format-specific refusal is a conversion guard, not a calibrated model-safety
guarantee or a replacement for the general experiment admission contract.

## Verification

All 74 scoped tests pass in 1.71 seconds, including 24 new parser/verifier tests.
The existing source and intake tests are included; the full legacy suite was
not rerun. The exact raw recount agrees with the saved audit. A separate CSV
implementation uses an ending-at-frame recurrence rather than the primary
run-cut counter and verifies every recording/class window table, including
histories 8/16/32/64, strides 1/12 and raw-frame horizons 10/25/50/100.

Analysis SHA256:
`52d789c7d2514029c58493904c96890e01f9d9be9634bb9845dd3c07257d6b99`.
See [machine-readable audit](analysis.json), [availability tables](availability.md)
and [separate verification](separate_verification.json). Separate implementation
here means arithmetic cross-checking in the same research workflow, not an
independent research-team replication.

```bash
.venv-pytorch/bin/python scripts/audit_m3w_traf_intake.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_traf_intake.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_traf_intake.py tests/test_m3w_traf_verification.py tests/test_m3w_external_source_audit.py tests/test_m3w_intake_admission.py
```

For a fresh output, run the audit without `--verify` and select a new versioned
`--output-dir`. The committed audit is not silently overwritten.

## Research Consequence

TRAF is not the shortest verified route to independent pedestrian scene support:
geometry, class semantics, physical-site/camera mapping, source-use conditions
and historical exposure all remain unresolved. This is not a negative forecasting
result; no predictor was evaluated on it.

DroneCrowd is a more relevant acquisition candidate, with a directly inspected
official raw-annotation download and explicit academic/non-commercial terms.
Its supplied validation set is sampled from its test set, so these folders must
not be used as independent selection/calibration/confirmation roles. The next
prerequisite is source/identity and camera-motion inspection, followed by a
user-approved role design before any predictive readout. See the
[source acquisition record](source_candidates.md).

The prior independent-scene versus strict-nested-development author decision
remains pending. This turn does not change it. Local disk has about 54 GiB free;
the audit needs no Torch/GPU or HPC. No new CREATE job or fresh remote queue
inspection is claimed. Stage5C and SMC remain off, and submission is not ready.
