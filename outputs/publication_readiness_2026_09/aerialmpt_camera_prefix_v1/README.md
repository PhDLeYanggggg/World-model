# Can dense aerial labels support forecasts from images?

I tested the frozen SDD aerial detector on the
[DLR AerialMPT release](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/aerialmpt-a-dataset-for-pedestrian-tracking-in-aerial-imagery).
Only four of the eight publisher-training sequences have at least
20 released images. I used the first eight images of each to make
twelve-step forecasts. The window starts were fixed from the
image lists, without filtering on any annotation outcome. No
publisher-test image or annotation member was opened.

I compared the original 2-by-2 tiles with a 4-by-4 grid, using
the same weights and non-maximum suppression. I ran the same
0.30 and 0.12 detector cutoffs and additive bridge as the
[SDD confidence study](../sdd_camera_support_v1/confidence_bridge_v2/README.md).
The bridge retains strong histories and requires at least three
high-confidence detections in an added eight-step history.
The release represents annotated pedestrian points as 4-by-4
markers; I matched detection centers one-to-one within 4 pixels,
and report 2- and 8-pixel tolerances separately in the readout.

| Input and threshold | Detection-to-label matches / 3,947 points | Online histories | Complete matched futures | Fully scored online multi-agent windows |
| --- | ---: | ---: | ---: | ---: |
| 2-by-2, 0.30 | 77 | 0 | 0 | 0 |
| 2-by-2, 0.12 | 448 | 4 | 2 | 0 |
| 2-by-2, additive bridge | 24 | 3 | 3 | 1 |
| 4-by-4, 0.30 | 7 | 0 | 0 | 0 |
| 4-by-4, 0.12 | 93 | 0 | 0 | 0 |
| 4-by-4, additive bridge | 0 | 0 | 0 | 0 |

The bridge count is for retained histories, not all detected
boxes. Its one-to-one matches are recomputed on that roster.
The one fully scored online window has two emitted forecasts
but 68 reference tracks with complete histories and futures:
it is not a fully observed crowd. There are 467 reference
histories and 415 complete reference futures overall.

## Why association alone is not enough

For each reference, I checked whether at least one box from
either grid, at score 0.12 or higher, lay within the allowed
distance in **every** observation frame. A box could be reused
for multiple references, and the online track ID could change
between frames. This is an upper bound for a no-gap tracker
restricted to those detections, not a forecast result.

| Match radius | Complete-future tracks with possible eight-step detection support / 415 |
| --- | ---: |
| 2 pixels | 0 |
| 4 pixels | 7 |
| 8 pixels | 16 |

At the primary tolerance, the 4-by-4 grid adds no possible
histories to the 2-by-2 grid. Better ID assignment alone
would not make the four windows adequate for risk calibration.
For this source, the next lever is a detector that can localize
small pedestrians, not another association threshold. A longer
source is still necessary for a recording-level comparison.
I did not train a policy or change the original 2% rule.

## Replay

Place the publisher ZIP at `external_data/AerialMPT_source_audit/publisher.zip`.
The [parent camera study](../sdd_camera_support_v1/README.md) describes the
pinned detector and local runtime. The first bank repeated exactly;
score and ceiling readouts also repeated. Inference took 48 seconds
after setup on arm64 CPU with four compute threads and no workers.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
py=.venv-pytorch/bin/python
"$py" scripts/run_m3w_aerialmpt_camera.py freeze
"$py" scripts/run_m3w_aerialmpt_camera.py score
"$py" scripts/audit_m3w_aerialmpt_detection_ceiling.py
```

[Registration](registration.json) | [Freeze](freeze.json) |
[Scores](readout.json) | [Detection support bound](detection_ceiling.json)

The result is development-only and in image pixels and released
image steps. It is not independent confirmation or a metric
forecast, and no reserved WEPDTOF or DroneCrowd outcome was opened.
