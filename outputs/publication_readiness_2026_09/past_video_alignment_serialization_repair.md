# Failed Record Serialization and Versioned Repair

The first local audit decoded all requested ETH/Hotel images but exited1 before
saving its local row records and aggregate report: crop-box coordinates were
NumPy int64 values, which the strict JSON writer rejected. Do not call this
original run complete. No model or metric was trained or selected.

Original runner/module/registration and decoded images remain under the ignored
`data/stage_cvpr2027_experiments/past_video_alignment/failed_source_snapshot` and
its parent. The original registration hash is retained; do not rewrite it as if
the first run succeeded. A v2 registration binds the repaired helper and this note.

Convert rounded crop coordinates explicitly to Python int before constructing
the rectangle, preserving the exact numeric request/clip behavior. A regression
test now serializes the returned record. Re-decode the short prefixes to a new
directory so the successful receipt traces actual source bytes, not unverified
partial files. This is a serialization repair, not an axis/mapping/model change.
