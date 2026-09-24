# Execution Record and Remaining Evidence Gaps

2026-09-24. All required local processes in this intake and its verification are
finished. This is real raw-data processing and validation, not a training result.

## Runs

| Task | Actual execution | Evidence |
|---|---|---|
| Official archive acquisition | PID 7905; complete; publisher MD5 and local SHA256 verified | V1 trajectory manifest |
| V1 pilot | PID 9676; complete; 188962 rows; 0.77 seconds recording work | V1 pilot.json |
| V1 full audit | PID 9715; failed at record 5, 4 completed | Missing class-name text field, preserved receipts |
| V2 full audit | PID 9915; all 376; 603.386 seconds | analysis.json |
| V2 full exact replay | PID 10613; all 376; 612.858 seconds | verification.json |
| Initial separate arithmetic | PID 11560; last recorded index 330; no completion artifact | Interrupted turn; later process absent |
| Repeated separate arithmetic | PID 12949; all 376; exit 0; 234.364 seconds | arithmetic_verification.json |
| Final combined scoped tests | 58 passed, 0.52 seconds | 27 raw-adapter + 6 grouping + 25 IMPTC regressions |
| Locality grouping | Fresh generation and identical rerun | site_groups_v1/site_groups.json |

The first arithmetic process is not credited as a completed run. Once its handle
and PID were absent and the completion receipt was missing, the same fixed
verifier was rerun fully. No completed raw audit or identity-bound results were
overwritten. Four V1 full-schema recording results also match the V2 summaries
exactly, excluding their necessarily different execution identities.

Peak raw-audit resident memory was 4,474,044,416 bytes. Runtime was native arm64
Python 3.11.1, NumPy 2.4.6 and pandas 3.0.3, single process. The ZIP was not
fully extracted; source data, track hashes and temporary dependencies remain in
ignored private directories. No Torch runtime or neural-training success is
claimed from this parser work. CREATE received only the separately recorded
read-only queue observation; existing simulation tasks were not modified.

## Evidence Identity

- Complete analysis SHA256: `db8c9c6e1ae4b0b21f663b30c2c372a0bd1c3d53248e02002332a80c6287e76d`.
- Summary SHA256: `f3c6527914f6a56cd358de5a6cdffefcd36a06d6b2c8d9a22778b97cb51e9208`.
- Independent arithmetic SHA256: `595bea1173ead17f088623453e45eec4af9e191519ab64ffc5644774a35cc1e2`.
- Group manifest SHA256: `16db623c8f956cfc92d73fa40c1766d0c8088af5312d703276186055d98d8feb`.

Both raw passes retain 152372066 rows, 514615 recording-scoped tracker IDs and
133078 short tracks. All 2256 prefix checks pass. Separate arithmetic uses
fixed, outcome-blind track/frame samples; it is not a second independent parser
for every numeric field or an online-tracking provenance certificate.

## Prioritized Next Work

1. Check partial recording aliases and related-source prior exposure, using the
   37 provisional locality groups as mandatory isolation keys. Resolve exact
   filename/statistics discrepancies explicitly without erasing source records.
2. Freeze data-role allocation and the external annotation-space input contract
   before looking at forecasting outcomes. Establish what the published tracker
   code can and cannot prove about temporal causality. Do not claim a calibrated
   ground plane or common physical-time horizon from raw pixel boxes.
3. Use admitted train-side data to test the rare-event-support hypothesis with
   unchanged risk rules and fixed controls. Reserve independent groups for
   calibration and confirmation; do not turn successful source parsing into a
   claimed prediction improvement.

The larger goal remains active and incomplete. Independent calibrated evidence,
the principal method claim, full comparisons and the submission package still
require work. There is no deployment change, Stage5C execution, SMC or foundation
claim. The new intake does not rehabilitate historical contaminated results.
