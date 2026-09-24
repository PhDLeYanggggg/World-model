# Partial-Clip Screen: No Cross-Locality Match Under Two Fixed Tests

2026-09-24. This is a data-identity result, not a forecasting result.

The full run verifies the official archive, freshly re-reads 375 recordings and
reuses the verified first-record pilot fingerprint. Every parsed row hash agrees
with the prior full raw audit: 376 recordings, 152,372,066 rows. It processes
8,653,298 nonempty frames and 8,421,334 consecutive eight-frame blocks.
The exact dynamic screen uses 8,421,186 blocks; the integer-pixel dynamic screen
uses 8,327,929. Static blocks are excluded from duplicate evidence only, not
deleted from the dataset. IDs and absolute frame offsets are not fingerprinted.

| Screen | Shared fingerprint groups | Recording pairs | Cross-locality pairs |
|---|---:|---:|---:|
| Exact raw class/box geometry | 0 | 0 | 0 |
| Integer-pixel geometry | 240 | 28 | 0 |

The full run completes in 320.92 seconds, peak RSS 3.97 GB. A separate matching
replay verifies every signature cache hash and recomputes both screens in 14.32
seconds. That second run is `cached_verified`, not another raw-data reparse.
The earlier raw audit already has its own complete fresh raw replay.

## Direct Review of All Candidates

All 240 quantized groups were separately checked against newly re-read raw
geometry: 27 recordings and 625 distinct eight-frame instances. Direct integer
geometry equality holds for every group. Every instance contains a single
detection in every frame. The maximum center excursion from its first frame
within these blocks is 0.970012 image pixels. These are near-static, low-precision
recurrences, not evidence by themselves of copied videos. No recordings are
dropped or relabelled. All matched recordings stay in their complete locality
group, regardless of the cause of recurrence.

## What This Does Not Establish

These tests cannot exclude re-encoded, rescaled, cropped, re-tracked or otherwise
approximately duplicated footage. They do not prove physical-site independence,
online tracking provenance or sensor accuracy. A negative exact test is not a
universal non-overlap certificate. No forecasting errors, goals, risk thresholds
or model choices were used. The follow-up role assignment retains these limits.

Ten overlap helper tests cover shifted clips, renumbered IDs, missing frames,
static blocks, frame order, integer sorting and future-prefix invariance. A
synthetic regression exposed the need to sort quantized geometry in its own
canonical order; it was fixed before the real pilot and frozen full run.

[Analysis](analysis.json), [cache replay](verification.json),
[direct candidate review](candidate_review.json), [registered scope](scope.md),
[role decision](../european_squares_roles_v1/conclusions.md).
