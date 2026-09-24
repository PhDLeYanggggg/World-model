# Source-Only History Cohort Built and Replayed

2026-09-24. Result source: fresh_run. Raw archive and frozen role dependencies
are cached_verified. No predictor fit or prediction-error readout is performed.

## Completed Cohort

All 163 admitted training recordings from 12 provisional locality groups are
processed under the fixed per-recording maximum256-query rule. The first real
recording pilot is verified and reused; the other162 are freshly built. A
separate complete replay freshly parses all163 raw recordings and reproduces
every input, label, index array and receipt exactly.

| Quantity | Count |
|---|---:|
| Selected recording/frame queries | 41,550 |
| Current-visible agent/query rows | 518,488 |
| Complete-past prediction targets | 318,969 |
| Incomplete-past context rows retained | 199,519 |
| Targets with all12 future labels | 193,705 |
| Targets with partial future labels | 118,217 |
| Targets with no future labels | 7,047 |
| Real-data future-truncation checks | 489 |

The source population is 31,111,802 past-eligible target windows; the318,969
targets are the complete registered subsampled cohort, not a full materialization
or a medium/full training claim. Overlapping queries are not independent units.
All visible agents remain together at a query, including short-history context.
Unknown future labels remain unknown, not zero-error/safe outcomes.

Cache arrays total493,003,312bytes and stay private. The ordinary build completes
in113.77seconds with peak RSS2.24GB; raw replay completes in115.40seconds with
peak RSS2.22GB. The pilot takes3.77seconds, mostly archive verification. These
are actual single-process data-processing timings, not neural throughput.

## Input Boundary and Verification

Inputs use unsmoothed raw box centers, observed masks, class_id, backward finite
differences, current visibility and the past-derived CV rollout. The registered
grid is8observed/12requested at raw stride12. There is no whole-track smoothing,
future interpolation, central velocity, fitted goal or endpoint-derived input.
Image-plane distances are not metric collision measures or common physical time.

Prediction targets require complete past support only. Future positions and
valid masks are separate label files, not feature fields. The full-data replay
also verifies the independently implemented past-support count against the
prior intake on every training recording. For three fixed queries per recording,
truncating all later raw rows leaves every input field exactly unchanged.
This is annotation-prefix causality; it does not prove the publisher's entire
sensor/detector execution provenance or human-ground-truth quality.

The training accessor rejects every reserved role. No model-selection,
calibration or confirmation recording is parsed into this training cohort.
The archive checksum still covers the complete acquired archive. All213
reserved recording outcomes remain closed.

79 scoped tests pass:58 prior intake/locality/IMPTC checks,10 overlap checks,
6 role/access checks and5 source-window checks. The new tests verify future
mutation/removal invariance, exact causal velocity, short-history neighbors,
unknown futures, role denial and deterministic grouping. This is not a rerun of
the entire legacy repository suite.

## What Changes Scientifically

There is now a reproducible, locality-separated training source on which to
test the previously observed support gap. This repairs a data-access and input
construction blocker, not the neural method. No new loss, ADE/FDE, gain, safety
calibration or model selection has occurred. Deployment remains unchanged.

Next: register a source-only, site-balanced predictor/risk-head experiment with
explicit source-internal cross-fitting. Measure whether gain/harm learning is
supported on these unsmoothed tracks before spending reserved model-selection,
calibration or confirmation evidence. Preserve the2%positive-easy and exact-zero
reference limits. New results must distinguish this oblique detector-track task
from SDD top-down annotation pixels. Stage5C and SMC remain disabled.

Manifest SHA256:
`6ed8f16f7f5557bbd88fbaa72778bb79b7893d6a91146945f0f76ced376d16a6`.
[Cohort manifest](analysis.json), [full raw replay](verification.json),
[role registration](../european_squares_roles_v1/registration.md),
[Chinese operation/recovery guide](operation_zh.md).
