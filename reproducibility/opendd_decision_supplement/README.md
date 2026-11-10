# Supplementary evidence: learning to allocate a fixed forecast budget

Run `python3 -I -B -S verify.py` after extracting. This uses Python's
standard library and no network, checkpoints, or trajectory labels.
It verifies the earlier matched-motion and external evidence in
`allocation_evidence.zip` before recomputing the newer learning controls.

All OpenDD rows use the same 606 development recording-seed views, 780
edits, nested paths, chronological credits and per-recording movement
caps. The offline allocator can see the complete score sequence and the
realized budget; it is not a streaming policy.

| Scoring objective | Original risk and positive gain over both references |
| --- | ---: |
| Frozen scores | 60 |
| Continuous gain and risk | 42 |
| Independent plan classification | 54 |
| Same-budget plan preference | 59 |
| Continued preference fit, equal-update control | 55 |
| Continued preference fit, risk-conditional utility | 58 |

In the last comparison, both arms start from the same checkpoints and
continue for 300 updates. In one stream only, the treatment compares two
donor plans that both pass the original risk checks but differ in gain.
Its paired conservative gain difference versus the continuation control
is -0.00197 percentage points (95% location interval [-0.01659, +0.01418]).
The three seeds are repeated views of each location, not independent sites.
The replay checks the aggregation and uncertainty; it does not rerun model
training or recreate the per-recording trajectory checks.
