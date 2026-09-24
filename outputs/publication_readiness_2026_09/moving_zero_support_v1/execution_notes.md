# Execution and Verification Record

Date: 2026-09-24. Status: full diagnostic completed, no model promotion.

## Executed Work

| Execution | PID | Observed completion | Elapsed seconds |
|---|---:|---|---:|
| Real coupa/seed17/Transformer pilot | 6224 | One view; raw cases checked | 60.467011 |
| Full matrix with resume | 6309 | 36 views, reusing one pilot view | 224.192550 |
| Complete verification replay | 6587 | All 36 views recomputed, exact result | 229.368053 |

Elapsed times come from the first source-verification event to the terminal event
in the local `events.jsonl`. The pilot neighborhood compute itself took 5.136262
seconds; source/provenance checks account for additional time. Verification,
separate arithmetic, tests and report generation are not folded into training
time. No training was performed. All three processes exited successfully and
their PIDs were absent in the completion check.

The runner used native arm64 `.venv-pytorch`, CPU threads 4, inter-op threads 1
and single-process data reading. This diagnosis fit local resources and did not
need CREATE. No SSH attempt, remote job submission, cancellation or simulation
project change was made for this diagnostic. The latest separately saved CREATE
attempt reported authentication denial; current remote assets/jobs remain unknown.

## Result Identity

`analysis.json` SHA256:

```text
470d5e393c7fa9a4496ca79dff405a4bdd935d921ca0038b308451893a80fbdc
```

The [replay receipt](replay.json) reports exact equality. Bound code, config,
upstream source identities and per-view hashes are checked by the runner.
It verifies all 175756 histories against their native coordinates and traces all
seven moving zero-reference windows to released raw annotations. All 36 views
and 126 rare-case feature/action/seed comparisons were retained; these are not
126 independent events. The cases represent three scoped tracks in two recordings.

## Separate Arithmetic and Tests

The [separate checker](separate_checks.json) uses SciPy `cdist` rather than the
diagnostic's chunked direct NumPy distance implementation. It checks all 36 view
hashes and all 126 rare-case comparisons. Neighbor ranks, zero-event counts,
benefit/harm counts and scoped-track counts match exactly. Maximum nearest-
distance difference is 4.163336342344337e-17. This is a separate arithmetic path
by the same executor, not independent scientific confirmation.

Checker SHA256:

```text
9238d815688493429d554c344a1b5ed9880c8852107a8b74345b3c976573e886
```

The scoped regression command in [operation/recovery](operation_zh.md) passed
35 tests, including nine new helper tests, in 0.80 seconds. The bound diagnostic
code did not change after these tests. The entire unchanged legacy suite was
not rerun. The aggregate SVG was rendered and visually checked; its local PNG
preview is not part of the public evidence package.

## Result Sources and Limits

| Item | Result source | Scope |
|---|---|---|
| Raw annotation and full-history checks | fresh_run | Supplied annotation coordinates, generated flags and control bracketing |
| Full neighborhood diagnosis | fresh_run | 36 views; one completed pilot view reused within the run |
| Complete replay and separate arithmetic | fresh_run | Exact result replay and independent implementation of distance arithmetic |
| Predictor outputs, forests, source rows and preprocessing | cached_verified | Upstream hashes/schema and outer-source provenance checked; not retrained |
| New neural/risk model or policy training | not_run | Diagnostic only; no new model or threshold authorized by these findings |
| External forecast readout | not_run | DroneCrowd confirmation remains closed; IMPTC remains quarantined |

The four SDD sites are development-exposed. Annotation control bracketing does
not reveal an actual annotation algorithm execution trace. Strict online
sensor-as-of causality is not established; code-level past-only inputs do not
resolve that source-provenance issue. Dense raw-frame errors are diagnostics,
not a change to the obs8/pred12 stride12 annotation-pixel primary metric.

No case is excluded because of its outcome, no risk limit is relaxed, no
deployment is changed and no safety guarantee is claimed. Stage5C and SMC remain
off. This diagnostic does not establish metric/seconds, true 3D, foundation
model or publication readiness. The research goal remains incomplete.

Only source, configuration and lightweight aggregate evidence are committed.
Private input arrays, full neighbor identities, checkpoints and third-party
annotations remain outside Git.
