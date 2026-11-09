# OpenDD: transport the calibration budget, not the numeric score

I kept the [frozen vehicle proposals and scores](../opendd_vehicle_fixed_release_v1/README.md)
and asked why the improved ranker still fails to release reliably. A
numeric score may not have the same scale at a new location. Applying
the calibration prefix's count or fraction to the held ranking removes
that scale dependence without fitting a threshold to held labels.

For each of 126 directions the calibration location supplies the
largest tie-complete prefix whose proposals all pass the original
full condition. On the held location, the count rule takes at most
that many records. The fraction rule takes at most the same fraction
of finite scores, rounded down. Neither splits a tied score group.
If the calibration prefix is empty, both abstain. Held labels are used
only to score these decisions. This is a batch diagnostic, not an
online quota or a replacement for the original 2% risk contract.

| Frozen score | Numeric: full / released | Same count: full / released | Same fraction: full / released |
| --- | ---: | ---: | ---: |
| Archived relative score | 1 / 14 | 0 / 9 | 0 / 5 |
| Direct full-condition score | 1 / 11 | 0 / 7 | 1 / 6 |
| Nested score | 0 / 0 | 0 / 6 | 0 / 2 |

These columns differ in coverage, so a smaller failure count is not
evidence of better utility. The direct count rule passes risk in 3/7
and the fraction rule in 2/6; neither gives a fully qualifying release
direction. Under an exact same-budget uniform control in each direction,
the direct count, fraction and numeric rules would find 0.36, 0.40 and
1.00 full passes in expectation. The fraction rule's paired
full-qualification gain is +0.00238 pp per available proposal
[-0.00304, +0.01051], not an established gain. Intervals use the same
3,000 resamples of seven location means.

## Where the ranking fails

The held bank contains 408 fully qualifying proposals in 118 of 126
directions. Every one of the direct head's **7 calibration-enabled**
directions has at least one full qualifying held proposal. But the
top tied score group is never entirely qualifying in those directions.
Three top groups already lack selected easy reference. Five top
scores also fall below the numeric calibration threshold; not one
of those five would give a clean top prefix if the cutoff were
lowered. The nested head's six below-cut directions likewise have
no qualifying top prefix, even though five contain a fully
qualifying candidate farther down.

A hindsight rule allowed to reorder the candidates could fill the
direct head's seven transferred count slots with qualifying proposals.
In contrast, a score threshold or any strictly monotone recalibration
can only take a prefix of the existing ordering. Neither can recover
those proposals without also admitting an earlier failure. This
explains why a better average top-five ranking is insufficient in
the actual calibration-enabled directions.

I retain CV fallback. The audit reads no raw trajectories, refits
no heads, alters no threshold of the archived policy, and opens no
independent Songdo or DroneCrowd outcomes. The 8,874 views repeat
493 recordings; the diagnosis is on exposed OpenDD development.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_vehicle_release_transport.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_release_transport.py tests/test_m3w_vehicle_release_score.py tests/test_m3w_vehicle_recording_replay.py
```

`fresh_run`: calibration budget transport and held prefix attribution.
`cached_verified`: the unchanged scores, local receipt hashes and
anonymous recording components. Eleven focused tests passed.
[Readout](readout.json).
