# A point detector for the small-person aerial case

The frozen box detector could recover only three complete camera
histories in the [AerialMPT image experiment](../aerialmpt_camera_prefix_v1/README.md).
I trained a small point detector on the source's own annotation
format to see how much of that gap came from the detector.

## Fit and evaluation

I use only the [DLR publisher training partition](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/aerialmpt-a-dataset-for-pedestrian-tracking-in-aerial-imagery).
The four original 8-to-12 windows stay unchanged. For each held
location, I fit on other locations and select a score threshold
using one further, disjoint location. The model takes one RGB
frame and predicts a half-resolution point score map. There are
15,313 trainable parameters. Training uses random 80-pixel crops,
axis flips, and a positive-weighted point loss. Every location-seed
fit runs 120 AdamW updates, with no held-location model selection.

The three fixed seeds are 17, 29, and 43, with nine fits and 1,080
updates in all. Score thresholds maximize point F1 within 4 pixels
on the calibration location. Detection peaks become fixed-size
association neighborhoods, not estimated person boxes. The same
online matching algorithm still requires eight consecutive frames
before forecasting. Each seed views the same four windows; neither
the seed replications nor the windows are independent locations.

| Detector | Matched observed points / 3,947 | Online histories | Complete futures / 415 | No scored future | Fully labeled joint windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen SDD detector with bridge | 24 | 3 | 3 | 0 | 1 |
| Frozen SDD detector, 2-by-2 at 0.12 | 448 | 4 | 2 | 2 | 0 |
| Point detector, seed 17 | 2,269 | 1,040 | 56 | 982 | 0 |
| Point detector, seed 29 | 2,035 | 717 | 56 | 658 | 0 |
| Point detector, seed 43 | 2,164 | 1,352 | 47 | 1,305 | 0 |

The bridge row counts matches for retained histories only. The
point detector rows include the per-frame tracker inputs. Complete
future counts are on the same reference windows for all rows.

Point detection increases the number of usable trajectories in every
seed. The detection-only eight-step upper bound rises to 124--141
complete-future references, compared with seven for the union of
the old detectors. The tracker realizes 47--56 of these. Most emitted
tracks still have no scored future. That is not a count of confirmed
false positives: it includes ambiguity in linking detections to
the same reference through all eight observations.

## Does an eight-step threshold fix this?

As a follow-up on the same development clips, I reused the fixed
weights and considered stricter thresholds on the original
calibration locations. Only the first eight observations and
their identity matches entered this choice. I required at least
two stable matches and at least 50% stable histories among emitted
tracks. No threshold cleared that screen for any of the nine
location-seed fits, so the guard emitted nothing. This is an
abstention outcome, not an accuracy or risk pass.

The point detector addresses part of the recall problem, but a
single score threshold does not resolve identity ambiguity. Before
using a camera stream for joint replacement, I would verify coverage
on longer recordings and separate localization quality from stable
tracking. These four short windows should not decide which forecast
policy to use.
[Guard calibration](track_guard/policy.json) |
[Guard readout](track_guard/readout.json)

## Replay

The local archive, detector-freeze protocol and input requirements
are described in the [parent study](../aerialmpt_camera_prefix_v1/README.md).
Native arm64 CPU, four Torch threads, and zero loader workers.
The per-fit checkpoint includes optimizer and patch RNG state;
rerunning `freeze` resumes every 40 updates and verifies the
frozen banks on a completed run. The fit writes a progress heartbeat.
The completed checkpoints and readouts replayed exactly.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
py=.venv-pytorch/bin/python
"$py" scripts/run_m3w_aerialmpt_point_detector.py freeze
"$py" scripts/run_m3w_aerialmpt_point_detector.py score
"$py" scripts/audit_m3w_aerialmpt_track_guard.py
```

[Location roles](registration.json) | [Model freeze](freeze.json) |
[Point detector results](readout.json)

This is a development experiment in released image steps and
pixels. It does not establish a gain for joint forecast replacement
or a safe deployment. No learned forecast policy, risk cutoff,
publisher-test label, or reserved confirmation outcome changed.
