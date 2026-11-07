# Causal Forecast Stream And Real Resume Check

I changed the prediction reader to slice the eight observed frames before
indexing agents. The earlier reader returned only past frames too, but its
advanced indexing copied a whole-track intermediate. The model interface and
target eligibility do not change. Every visible agent stays in the query
inventory; the existing model requires a complete observed history for its
targets and selected neighbors.

| Check | Evidence |
|---|---|
| Old vs bounded input | All arrays and packed geometry exact on the original 336 queries in 112 recordings, with 48,183 eligible targets |
| Complete unassigned input roster | 112 recordings, 31,472 queries, 4,558,096 visible and 4,520,912 past-complete instances, 35,375 batches; no zero-target queries |
| Fixed Torch predictor resume on exposed A | PCT and EqMotion, each with seeds 17/29/43; 2,304 saved rows reused after process exit, 3,840 additional rows computed; all 6,144 match the original bank exactly |
| Compute | Complete packing 79.77 s / 300,154,880 B peak RSS; Torch resume 8.10 s / 753,811,456 B, native arm64 CPU4, one inter-op thread, zero loader workers |

The input scan retains only per-recording counts and SHA256 digests of ordered
keys and geometry. It does not write dense features. Each query key retains
frame and agent identity. A synthetic test also traps any attempt to copy a
full time axis in the new reader. The old implementation is read for a matched
comparison on the 336 probes; it still does not supply future fields to the
model. Source-file hash checks cover complete stored arrays, not future targets
materialized for training or scoring. The forecast-resume probe uses only old
development geometry and cached predictions, never DroneCrowd predictors or labels.

## Resource Decision Before A New Readout

| Existing, unassigned alternative | Groups | Clips | Queries | Past-complete rows |
|---|---:|---:|---:|---:|
| Split proposal: calibration | 15 | 40 | 11,240 | 1,757,584 |
| Split proposal: confirmation | 30 | 72 | 20,232 | 2,763,328 |
| Frozen zero-shot proposal: all confirmation | 45 | 112 | 31,472 | 4,520,912 |

Neither alternative is assigned by this inventory. Counts use only past support;
they do not promise nonempty easy references or independent sample sizes.
The small A resume measured about 258 EqMotion rows/s, implying 4.86 hours per
checkpoint over the full roster before scoring. PCT was about 3,500 rows/s,
0.36 hours per checkpoint including setup in this short probe. These are local
projections, not measured external forward timings. Three full EqMotion passes
alone project about 14.6 hours serially. A packed float32 geometry table would
take 8.61 GB. The six raw key-and-path banks together would take at least 3.04
GB before containers and manifests. On the current disk keep only recording-
scoped forecast buffers and verify each completed unit before reclaiming it.
The largest recording has 115,861 target rows (about 78 MB of raw key/path data
for six checkpoints). There is no need to downsample to fit storage.

The fresh CREATE read-only query timed out after 60 s; remote status is unknown
and no job was submitted. Local development and the full input scan completed.
For real multi-seed inference, use CREATE only after verifying data placement,
queue and resume state; the local alternative is consecutive resumable passes.

[Chinese operations](operation_zh.md) | [Full input schedule](schedule_analysis.json)

No external forecaster forward or future-label method, calibration, scientific
role assignment, new training, independent result or deployment. Pixel/native
8/12, not seconds or metric. Original risk/support/NA rules remain, no Stage5C
or SMC. The paper remains a working draft.
