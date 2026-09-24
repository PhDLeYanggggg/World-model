# European Squares Raw Intake: Verified Structure, No Model Claim

2026-09-24. Result source: fresh_run. Official bytes are cached_verified against the pinned acquisition manifest.

## What Completed

All 376 released raw CSV recordings were read and replayed, covering 39 nominal square IDs,
152,372,066 rows and 514,615 recording-scoped tracker IDs. These IDs are not a count of unique people.
No raw recordings are omitted because of short history, missing future support or forecast difficulty.
133,078 tracks shorter than 30 observations remain. Query eligibility uses past support only.

2,256 prefix mutation/truncation checks pass. Complete re-parsing reproduces all recording arrays and summaries.
Separate arithmetic replays all rows and checks 362,760 fixed past-membership cases,
9,024 complete future-count cases and 3,104 direct velocities.
The separate counting covers fixed samples of up to three tracks per recording, not every track.
52 scoped tests pass. This is not a rerun of the full legacy test suite.

## Past Support and Label Availability

Stride is an index increment in the released raw detector frames. These probes do not establish common seconds across datasets.
Counts overlap in time and across history lengths. They are not independent experimental sample sizes.

| History/grid | Past eligible | All 12 future labels | Partial future | No future | Recording/frame queries |
|---|---:|---:|---:|---:|---:|
| K16_stride1 | 120,734,583 | 107,576,074 | 12,769,702 | 388,807 | 8,201,886 |
| K16_stride12 | 63,005,268 | 41,189,937 | 20,933,232 | 882,099 | 7,044,173 |
| K32_stride1 | 104,015,741 | 94,887,688 | 8,870,352 | 257,701 | 7,930,006 |
| K32_stride12 | 36,414,191 | 26,101,699 | 9,926,283 | 386,209 | 6,046,760 |
| K64_stride1 | 83,048,517 | 77,301,210 | 5,588,479 | 158,828 | 7,537,462 |
| K64_stride12 | 16,567,328 | 13,344,076 | 3,114,345 | 108,907 | 4,617,444 |
| K8_stride1 | 133,295,590 | 115,837,012 | 16,905,516 | 553,062 | 8,389,013 |
| K8_stride12 | 90,701,631 | 54,070,558 | 35,021,410 | 1,609,663 | 7,702,971 |

## Limits That Remain

The metadata/archive discrepancies and the initial schema failure are retained in [versioned_repair.md](versioned_repair.md).
Exact full-track geometry screening found 0 cross-recording groups and
0 within-recording extra aliases. This screen does not exclude partial clip duplicates or shared cameras.
Physical locality, related-source exposure, recording correspondence, publisher online processing and frame-time mappings still require admission decisions.
All labels are automated. No verified metric transform or human-gold claim is made.
The seasonal collection repeats a few locations; it cannot be counted as hundreds of independent scenes.

The source remains quarantined and unassigned: zero sites are yet admitted to independent calibration or confirmation.
No baseline or neural forecast errors were opened, no goals were built, no model was trained and deployment is unchanged.
DroneCrowd remains closed confirmation; the exposed SDD sites remain development-only. Stage5C and SMC are not executed.

## Next Evidence Step

Reconcile recording/site identities and screening across related sources, then freeze a conservative site-group role manifest.
Specify the raw point convention, history grid and automated-label provenance without claiming online sensor or metric validation.
Only after role admission should train-side support and a source-only method test be opened; independent confirmation must not select models.

[Source provenance](../european_squares_intake_v1/provenance.md), [machine summary](summary.json),
[replay](verification.json), [separate arithmetic](arithmetic_verification.json), [operation/recovery](operation_zh.md).
