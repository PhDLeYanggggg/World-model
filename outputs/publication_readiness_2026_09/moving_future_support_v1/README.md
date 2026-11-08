# Where the missing MovingDroneCrowd++ labels come from

**Result: no exact label can be restored from the verified release.**
This audit reopens only the original 24 fitting recordings from nine
source groups. A separate CSV reader enumerates contiguous eight-step
histories and performs exact lookup by recording, track ID and future
annotation step. It compares the resulting roster, relative head centers
and validity masks with the hash-bound frozen fitting arrays. It never
fills gaps, links track IDs across clips, or relabels a window.

| Check | Count |
|---|---:|
| Development windows | 69,746 |
| Requested future steps | 836,952 |
| Released matching labels | 685,547 |
| Missing step occurrences | 151,405 |
| Distinct missing recording/track/frame keys | 29,268 |
| Missing steps after the last observation of the same ID | 132,376 |
| Missing steps inside that ID's recorded span | 19,029 |
| Mask / known-coordinate / past-alignment mismatches | 0 / 0 / 0 |
| Exact raw labels omitted by the converter | 0 |

Of 1,892 windows with no released future label, 1,812 have no later
record for the same ID. The remaining 80 reappear after the prediction
window. Splitting the original CV/OLS4 disagreement floor by these
events reproduces **11.551 pp**, of which **11.474 pp (99.33%)** comes
from windows without a later same-ID record. It does not prove those
people exited physically. The missing support is mostly not a short
gap that can be bridged from a later point, and interpolation would
not create verified labels in either case.

The [publisher's project](https://github.com/fyw1999/MovingDroneCrowd)
documents a MOT CSV per clip. The current publisher metadata matches
the pinned [data revision](https://huggingface.co/datasets/fyw1999/MovingDroneCrowd/tree/0867fef0a8885fceb4253bbeab394f3d4dd85f5c):
one data archive, a conference scene-label file, the dataset card and
Git attributes. The ICCV GitHub release lists no separate attached
annotation asset. This verifies the checked public channels, not
the nonexistence of unpublished or future labels.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_moving_future_support.py
```

A second fresh raw parse reproduced the registered summary. No changes
to the annotation, split, role, horizon, scoring rule or confirmation
seal. Related regression suite: 47 passed.

[Audit](analysis.json) | [OpenDD control](../opendd_weighted_velocity_v1/README.md).
