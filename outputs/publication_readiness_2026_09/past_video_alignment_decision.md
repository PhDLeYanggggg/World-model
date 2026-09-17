# Past-Video Availability and Axis Check

## Material Passport

Local, fit-only media inspection. No forecasting, body-orientation model,
calibration, threshold selection, new scientific role or visual-training approval.
Parent8/12 native-step task, labels and primary metric remain unchanged.

Local upstream `ui_trajlets.py` and `play.py` draw H-inverse outputs in reversed
axis order: OpenCV(x,y)=(projected[1],projected[0]). The first stationary scene
availability audit interpreted those outputs directly as(x,y). Compare both
conventions numerically and preserve the original bound report. No current
forecast uses image pixels; static world-coordinate XML features are not changed.

The upstream loader derives ETH effective fps from annotation spacing times2.5,
and its video helper uses15 for ETH,25 for Hotel. Both actual video streams report
25 via the decoder. This reinforces an encoding/source clock ambiguity; it is
not evidence allowing a seconds-level claim. The playback loop indexes frame t
directly, without a time conversion. Inspect that declared mapping, do not fit a
temporal offset or choose a different speed against downstream error.

Select the first frozen stationary query per agent, sorted by recording, agent
and frame. Retain all31 agents, independent of any future change label. Request
the first and last images from each query's eight observed native-frame indices.
Project only its current/past coordinates using supplied H. Decode sequentially
by frame index, storing selected frames and fixed96x96 inspection crops locally.
No future query endpoint, supplied velocity, destination/group label or still
background image supplies an inference feature. No crop is a verified body box.

Report source hashes, codecs, rates, frame requests, coordinate bounds, clipping
and decode status. View representative raw-frame/crop samples locally. Qualitative
inspection cannot certify all agents, actual capture-time causality, annotation
construction or metric geometry. No imagery, raw rows or decoder binaries go
to GitHub; only code, configuration and aggregate findings are public.

PyAV18.1.0 is installed in ignored `data/stage_cvpr2027_experiments/media_decode_runtime`,
an isolated arm64 wheel. The existing PyTorch environment was not modified.
Visual model training remains not_run unless source-use and as-of alignment
requirements are met under an explicitly recorded future experiment.
