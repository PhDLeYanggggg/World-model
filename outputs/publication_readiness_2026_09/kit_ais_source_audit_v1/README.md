# KIT AIS: timing and whole-roster support

I checked a source with both aerial images and published detections, so
the availability of manually labeled trajectories would not be mistaken
for the availability of online tracking inputs. I obtained the
[KIT AIS](https://www.ipf.kit.edu/downloads_People_Tracking.php) publisher
training archive and detection archive (22.1 MB combined). The publisher
lists [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) and asks for
attribution to the KIT AIS Data Set. No evaluation archive was downloaded;
only the seven training XML members were read from each archive. No raw
assets are in Git.

## What the source supports

All seven detection timelines match the reference XML's frame number,
image name and UTC timestamp. The raw image names are in the training
archive. The seven clips have 24, 18, 14, 12, 19, 13 and 4 release frames.
`AA_Easy_01`, `AA_Easy_02` and `AA_Walking_01` share 10, 10 and 12
timestamps pairwise. They cannot be used as independent recording groups.
Different names are not evidence of different sites.

Only `Munich01` is long enough to request eight observed and twelve future
positions by release order. Its median timestamp gap is 0.400 s, but it has
a 0.799 s gap and splits into runs of 13 and 11 frames when gaps over 1.5
times the median are marked. This is a timing diagnostic, not a changed
forecast cadence. No seconds-level forecasting or metric accuracy claim is
made. The source provides nominal ground sample distance; the raster
alignment is not calibrated by this check.

| Fixed detection score cutoff | Online eight-frame histories | Stable matches | Complete labeled futures | Multi-agent windows with all online futures |
|---:|---:|---:|---:|---:|
| 0.00 | 41 | 9 | 7 | 0 / 1 |
| 0.60 | 3 | 2 | 2 | 0 / 1 |
| 0.75 | 0 | 0 | 0 | Not estimable |

The same first 20 release positions are used in all rows; they are not
three windows. There are 36 complete reference trajectories in that
window. I discard publisher detection IDs, track only the first eight
frames, and freeze those outputs before joining reference boxes. A match
must retain the same reference identity throughout the eight observations.
The three cutoffs are diagnostics; none is selected from outcomes. The
detector's original fitting exposure is unknown, and the supplied image
mosaics are not verified live sensor inputs.

## Next source

KIT AIS is useful for reproducing the input-support failure, not for
calibration or independent confirmation. The publisher lists only one
evaluation recording with at least 20 release frames; its outcomes remain
unopened. The official [VisDrone-MOT](https://github.com/VisDrone/VisDrone-Dataset#task-4-multi-object-tracking)
validation and test-dev Google Drive downloads returned quota pages, not
ZIP bytes. An available official route and a detector with disjoint
training provenance are needed before assigning any new roles.
[Okutama-Action](https://github.com/miquelmarti/Okutama-Action/blob/master/index.md)
is another lead, but paired drone views must stay grouped and only its
tracking-label variant has persistent IDs. It was not acquired.

## Replay

```sh
.venv-pytorch/bin/python scripts/audit_m3w_kit_ais.py --download
.venv-pytorch/bin/python scripts/audit_m3w_kit_ais.py --verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_kit_ais.py tests/test_m3w_ht21_detection_support.py
```

`fresh_run`: archive acquisition, clock and roster checks. `cached_verified`:
pinned archive bytes and byte-identical replay. Thirteen scoped tests pass.
`not_run`: detector fitting, forecast scoring, risk calibration or
confirmation. No reserved outcomes or deployment changes.
