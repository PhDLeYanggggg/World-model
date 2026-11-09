# VisDrone-MOT: label support versus camera-track support

I checked whether VisDrone-MOT could support the same eight-observation,
twelve-prediction comparison used elsewhere in this project. The labels
look promising. The current camera input does not yet recover enough of
the group to test a joint forecasting decision.

## Source and label coverage

The [VisDrone release page](https://github.com/VisDrone/VisDrone-Dataset#task-4-multi-object-tracking)
lists the training ZIP. The official download returned a quota page in the
earlier source check. Two pinned mirrors declare the same 8.08 GB archive
and LFS digest. I fetched 13.50 MB in byte ranges for intake: the ZIP
directories matched, three sampled annotation members matched byte for
byte, and one JPEG decoded. This is a partial integrity check, not a
full-archive hash or a verification against the publisher's own bytes.
[Mirror revisions and hashes](source_receipt.json).

The archive contains 56 clips, 24,201 contiguous numbered JPEGs, 1,169,359
annotation rows and 234,305 included pedestrian boxes. The frame count
matches the [publisher toolkit](https://github.com/VisDrone/VisDrone2018-MOT-toolkit);
the [2019 challenge paper](https://openaccess.thecvf.com/content_ICCVW_2019/html/VISDrone/Wen_VisDrone-MOT2019_The_Vision_Meets_Drone_Multiple_Object_Tracking_Challenge_Results_ICCVW_2019_paper.html)
reports 24,198. I retained the observed count. The filename stems yield
48 candidate recording groups, not 48 verified independent sites.

| Sampling by release frames | Nonoverlapping windows | Complete pedestrian histories and futures | Windows with at least two complete references |
| --- | ---: | ---: | ---: |
| Every frame | 1,185 | 10,433 | 934 |
| Every fourth frame | 276 | 1,911 | 191 |

At stride 4, the 191 multi-agent windows come from 47 clips. These are
counts from labels, not counts from the camera tracker. No physical
scale or frame-to-seconds conversion is verified. [Per-clip counts](readout.json).

## Camera pilot

I used the first eligible clip of each of the first four distinct filename
groups, starting at frame 1 and reading only the eight past JPEGs at stride 4.
The existing COCO-pretrained detector and causal tracker ran unchanged on
native arm64 CPU, using four Torch threads and no workers. I froze the four
prefixes before the pilot matched them to any reference rows. The future
labels were used only to count support, not to tune or score a forecast.

| Reference scope | Complete references | Detected in all 8 past frames, ignoring identity | Online histories | Online matches with all 12 future labels | Fully labeled online rosters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pedestrian | 76 | 39 | 53 | 14 | 0/4 |
| Pedestrian plus other people | 119 | 45 | 53 | 17 | 0/4 |

Three windows contain at least two fully matched pedestrian histories.
None supports an error comparison on the whole online roster. Adding the
other-people category accounts for little of the gap. Both detection coverage
and stable association need work; changing the forecast head would not fix
this input problem. The four clips are a diagnostic, not a representative
estimate for VisDrone. [Registration](camera_registration.json),
[frozen prefixes](camera_freeze.json), [strict counts](camera_support.json),
and [label scope sensitivity](camera_label_scope.json).

## Replay and next experiment

```sh
.venv-pytorch/bin/python scripts/audit_m3w_visdrone_mot_train.py --download
.venv-pytorch/bin/python scripts/audit_m3w_visdrone_camera_pilot.py
.venv-pytorch/bin/python scripts/audit_m3w_visdrone_mot_train.py --verify
.venv-pytorch/bin/python scripts/audit_m3w_visdrone_camera_pilot.py --verify
.venv-pytorch/bin/python scripts/audit_m3w_visdrone_label_scope.py
.venv-pytorch/bin/python scripts/audit_m3w_visdrone_label_scope.py --verify
```

The pilot uses the local Torchvision weights bound in the registration.
The verification commands recompute the counts and all 32 image prefixes
offline from the hashed local sources. Only code and aggregate results are
in Git. The archive and checkpoint are not redistributed.

First compare a causal association improvement on identical past detections
with a detector improvement on the same training-only inputs. Measure stable
identities and whole-roster support before spending compute on forecasting.
Then verify recording and site provenance, source terms, and a held-out split
before using VisDrone for a paper claim. Neither validation nor test payloads
were opened. The existing manuscript, confirmation data, policies and risk
rule are unchanged.
