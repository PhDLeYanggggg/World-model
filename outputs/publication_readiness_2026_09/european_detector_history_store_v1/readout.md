# Verified Past Detector Store

Result source: fresh_run raw parsing/alignment, followed by a separate fresh_run
raw reparse. Forecast scores and new model training are not_run in this store
report. The originally failed startup remains in pilot_attempt_v1.json.

## Completed Material

All 74 admitted fitting recordings and 149,510 original packed queries are
present. Every query has eight original unsmoothed boxes and confidence values
at raw frames q-84, q-72, ..., q. Exact checks retain the original agent/frame,
box coordinates, midpoint history and current width. The 222 actual prefix
truncation checks reproduce the same past features. There is no future-target
array access, score-based row removal, forecast-target change or split change.

The 64 ordered feature names and all private array hashes are in complete.json.
They comprise eight values each for log1p(width), log1p(height), confidence,
log aspect, normalized half-height changes, normalized midpoint dx/dy and
successive box IoU. Zero-area boxes remain finite with the registered one-pixel
regularizer; they are not silently removed. Sensor-acquisition-as-of is not
certified for this supplied tracker export.

| Execution | PID | Seconds | Peak RSS bytes | Raw records parsed | Cached records verified |
|---|---:|---:|---:|---:|---:|
| Full extraction | 39447 | 40.495374 | 1,989,165,056 | 73 | 1 |
| Separate complete reparse | 39736 | 39.160109 | 2,041,085,952 | 74 | 0 |

Both processes terminate with exit 0. The separate verification reports all
arrays exactly equal. Final NPY arrays total 52,030,248 bytes; with per-record
recovery checkpoints the local subtree is approximately 92 MB. Raw input and
private arrays are not uploaded to GitHub. Confidence min/p10/p50/p90/max are
0.250001/0.473782/0.792745/0.888068/0.964906; these are descriptive store
statistics, not a normalizer fitted across experimental contexts.

## Evidence Boundary

This resolves a concrete input-alignment gap, not the failed forecasting or
complete-policy risk gate. A matched model experiment is still required.
Independent selection, calibration, confirmation, held readout and DroneCrowd
remain closed. SOURCE is supplied detector-silver midpoint pixels,
obs8/pred12/stride12 raw frames, not metric or seconds or human gold. No
deployment change, Stage5C execution, SMC, true-3D or foundation claim.
