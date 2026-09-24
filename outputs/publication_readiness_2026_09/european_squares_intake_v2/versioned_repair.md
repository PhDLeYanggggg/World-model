# Explicit Schema Repair and Archive Reconciliation

2026-09-24. Result source: fresh_run. No forecasts or fitting.

## Preserved Failure

The v1 whole-archive audit completed four recordings, then stopped at Bevagna's
morning recording because its raw export omitted `class_name`. The original
parser, code identity, successful pilot and four private recording receipts
remain unchanged. This was an over-strict input contract, not bad coordinates
or a reason to delete the recording.

A header scan of all 376 original CSV members finds exactly two variants:
316 contain the nine documented fields; 60 contain the same numeric fields
without the text class name. V2 accepts only these two explicitly observed
schemas. Missing text stays `not_provided`; numeric class IDs, boxes, confidence,
agent IDs and frame IDs are unchanged. No other required field is optional.
Duplicate identities, nonfinite values and invalid geometry remain errors.

V2 starts a separate output namespace and recalculates all recordings. It does
not relabel v1's interrupted audit as complete or reuse differently bound
recording results. 52 scoped tests pass, including seven new schema regressions,
20 earlier intake tests and 25 existing IMPTC tests. The full legacy suite is
not rerun for this source adapter.

## Counts Are Not All Recordings

The verified archive SHA256 is
`068e584db8895447516d548de7cd3a7b67b1ceaae16afdf76c677602e5e8235f`.
Its raw trajectory members total 10,414,642,404 uncompressed bytes, read one
recording at a time. The entire archive is not extracted.

There are 147 comparative and 229 seasonal raw CSVs, covering all 39 nominal
square IDs. The published seasonal statistics have 244 rows, but only 236
distinct `(city, date, timeslot)` keys: eight keys are repeated. Of these unique
keys, seven lack a matching raw member. This accounts for the 15-row difference;
it is not evidence that 15 unique raw videos were deleted by our reader.

All 229 raw seasonal files have a name/date/slot match in the table. Five
available Biberach records are tagged with square ID 9 in one metadata row even
though their original path and filename use ID 8. Two other ID-9 Biberach rows
have no raw match. The exact differences are retained in
[archive_reconciliation.json](archive_reconciliation.json).
No processed file is substituted for absent raw support and no source metadata
is silently corrected. A later role manifest must record any adopted mapping.

These are source inventory findings, not independent calibration admission,
forecast improvement or proof of causal tracking. The new source remains
quarantined and unassigned until the complete structural and provenance work is
reviewed. DroneCrowd confirmation remains closed.
