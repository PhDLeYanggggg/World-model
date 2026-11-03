# Small Publisher Utilities Do Not Calibrate Our Sampling Clock

The pinned author's [info.json](https://github.com/fyw1999/MovingDroneCrowd/blob/df2b44d788b1f4bbb270139c52247ae68f699130/cusdatasets/dataset_prepare/info.json)
contains634integer-valued entries and no identifier matching the120clip IDs
in our pinned archive. No explicit FPS, frame-rate or timestamp key appears.
Fetched payload:16,326bytes, SHA256
7df40ebcb3dd8b5ee716b208c7cb593856dc4d83e0d854d2dd81c0e7cf0ff2f7.

The nearby [scene-label utility](https://github.com/fyw1999/MovingDroneCrowd/blob/df2b44d788b1f4bbb270139c52247ae68f699130/cusdatasets/dataset_prepare/scene_label.py)
uses those integers for density-bin assignments in a different filename
namespace. It is not per-video FPS evidence for this release. The code is
read as source, not executed or copied into our runtime.

Therefore this inspected metadata does not change effective_seconds=None.
This is a limited source check, not proof that timing information exists
nowhere. Native annotation+50/+100 remain distinct from verified raw-video
frames or physical seconds. No calibration is inferred from directory names,
density codes, total frame counts or descriptive duration strings.
