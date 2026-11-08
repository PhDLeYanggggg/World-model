# Waterloo Map Registration and Pedestrian Support

I checked whether the public `DJI_0932.MP4` and `932.db` pair could
support an independent, raw-video pedestrian forecasting experiment.
The results below are fresh engineering measurements, not forecast
accuracy or an independent scientific confirmation.

I matched the winter video to the
[2020 Waterloo municipal orthophoto](https://gis.waterloo.ca/maps/rest/services/Public/Public_Imagery_Archive/MapServer/11),
with the [precise grid request and digest](analysis.json). Four road
markings were withheld from image fitting and checked separately.
The first-frame bridge fits current images to the already-observed
first image, not to any later frame. Trajectory labels did not enter
either image fit. The fixed current-frame label comparison used the
nominal clock and a three-map-unit matching radius.

| Diagnostic | All sampled geometry | Spaced keyframes | Early geometry | Person boxes | Person label observations | Person links | All links |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct frame-to-map, 4 by 3 tiles | 7/19 | 2/12 | 6/8 | 7 | 24 | 0 | 29 |
| First-frame bridge, 4 by 3 tiles | 10/19 | 3/12 | 8/8 | 10 | 32 | 0 | 41 |
| Same bridge, 8 by 5 tiles | Prefix only | Not run | 8/8 | 76 | 32 | 0 | 35 |

The first two rows do not share identical label support; the last two
do. Counts refer to repeated frame observations, not distinct agents.
The raw-to-first fit itself passed its feature-match criteria at the
late keyframes; the map holdouts failed. That is why a visually plausible
stabilization is not enough to claim whole-clip ground alignment.
For the coarse bridge, median and 95th-percentile distance among accepted
links were 0.369 and 2.328 map-grid units; no accepted link was pedestrian.
Smaller tiles raised detector output but did not recover the annotated
pedestrians. This does not distinguish detector false positives, export
datum, object anchor or label alignment by itself.

**Decision:** no forecast scoring, metric claim, timing claim or scientific
role for this video. Exact annotation anchor, datum and synchronization
are not established. An independent pixel-to-label correspondence check
on an engineering sample, or a better-documented annotated video source,
is more informative than another detector-threshold search.

Reproduction (publisher assets and the government raster are local and
not included in Git):

```sh
.venv-pytorch/bin/python scripts/audit_m3w_waterloo_ortho.py
.venv-pytorch/bin/python scripts/audit_m3w_waterloo_ortho.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_waterloo_ortho_tracking.py
.venv-pytorch/bin/python scripts/audit_m3w_waterloo_person_tiles.py
.venv-pytorch/bin/python scripts/audit_m3w_waterloo_person_tiles.py --replay
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_ground_plane_registration.py tests/test_m3w_waterloo_intake.py tests/test_m3w_streaming_boxes.py
```

[Direct protocol](protocol.md) | [Bridge protocol](bridge_protocol.md) |
[Tile protocol](tile_protocol.md) | [Direct output](analysis.json) |
[Bridge output](causal_bridge.json) | [Tile output](fine_tiles.json).
The prior DroneCrowd calibration result remains a failure under the
original 2% rule; 30 confirmation groups remain sealed. No Stage5C or SMC.
