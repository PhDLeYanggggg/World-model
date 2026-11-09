# Do weaker detections recover usable camera histories?

I reran the aerial detector on the same 92 development windows in
23 SDD training clips. The video pixels, sampled frames, detector
weights and raster registration did not change. The question was
whether lower-confidence observations could fill gaps in eight-frame
histories, without replacing the strong tracks already available.

I compared the original 0.30 detector threshold, a 0.12 threshold,
and an additive bridge. The bridge keeps the original tracks and
forecasts exactly. It only adds an eight-frame low-threshold track
if at least three detections also meet the original threshold and the
track does not duplicate an existing history. This is a development
diagnostic, not a new tracking method or an implementation of
[ByteTrack](https://arxiv.org/abs/2110.06864).

## Where did the unscored tracks come from?

On the original detector roster, 46 of 126 histories had all 12
pedestrian outcome steps. Of the 80 with no scored outcome, 37
matched a non-pedestrian label in all eight observation frames:
31 matched cyclists and six matched skaters. Two matched a
pedestrian intermittently, five matched another class intermittently,
and 36 never matched a visible label. The last group is unresolved,
not a count of false detections. The audit explains the original
denominator; it does not change the pedestrian scoring rule.
[Class audit](../raster_registration/unknown_taxonomy.json).
The [follow-up](../class_support_v2/README.md) scores the additional
classes separately and counts fully labeled joint windows.

## How much useful support did the extra tracks add?

| Camera input | Online histories | Complete pedestrian futures | Wholly unknown | Clips with adequate support |
| --- | ---: | ---: | ---: | ---: |
| Original threshold | 126 | 46 (36.5%) | 80 | 1/23 |
| Lower threshold | 423 | 109 (25.8%) | 312 | 3/23 |
| Additive bridge | 237 | 82 (34.6%) | 154 | 2/23 |

The bridge gives more labels in absolute terms, not a higher rate
of usable labels. It adds one supporting clip, `hyang/video4`, to
the original `hyang/video5`. Lowering the threshold outright adds
`coupa/video1` as well, but raises the number of unscored histories
from 80 to 312. Of the 111 histories added by the bridge, 36 have
complete futures, one has a partial future, and 74 are wholly
unknown. No original matched identity changes. These are repeated
windows within development clips, not independent confirmation.
Even on the bridge, only 17 of 992 visible future positions lack
the publisher's generated flag; no complete future consists entirely
of such positions.

## Does the motion gain survive on the full roster?

| Camera input | OLS8 vs. OLS4 on known steps (pp) | OLS8 vs. OLS4 lower bound (pp) |
| --- | ---: | ---: |
| Original threshold | +1.83 | -15.48 |
| Lower threshold | +2.66 | -53.68 |
| Additive bridge | +3.66 | -22.10 |

Each row compares the two motion forecasts on its own full online
roster. I weight nonempty windows equally and subtract the forecasts'
disagreement at missing steps, normalized by observed constant-velocity
error. The rows do not compare the same sets of agents. The negative
bounds do not establish harm; they mean I cannot establish a gain
for all forecasted tracks. I did not fit a policy or change deployment.
For another development pass, I would prioritize matched outcomes
across more scenes and separate non-pedestrian supervision before
trying lower thresholds again.

[Registration](registration.json) | [Freeze](freeze.json) |
[Coverage readout](readout.json) | [Full-roster utility](full_roster_utility.json)

## Replay

The [parent experiment](../README.md) describes the local video,
annotation and weight requirements. The freeze checks the exact
decoded observation pixels and reproduces the original high-threshold
tracked boxes and forecasts on every window. It repeats the first
window, checkpoints each window and writes a heartbeat with PID and
progress. Labels are read in the separate scoring commands.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
py=.venv-pytorch/bin/python
"$py" scripts/run_m3w_sdd_camera_confidence.py freeze
"$py" scripts/run_m3w_sdd_camera_confidence.py audit
"$py" scripts/audit_m3w_sdd_camera_unknown.py
"$py" scripts/audit_m3w_sdd_camera_confidence_utility.py
```

The first inference pass completed, but its scoring process failed to
serialize a NumPy integer. I fixed the counter conversion and reran
inference under a new registration, without changing the thresholds
or evaluation. All 92 low-threshold observation banks agreed exactly
across the two runs, and coverage and class readouts replayed exactly.
No external confirmation outcomes were reopened.
All distances are video pixels and the windows are measured in raw
frames, not meters or seconds. The original 2% risk rule is unchanged.
