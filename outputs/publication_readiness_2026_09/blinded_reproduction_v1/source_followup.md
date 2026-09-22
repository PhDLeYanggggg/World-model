# TRAF Primary-Source Geometry Follow-up

## Material Passport

Fresh primary-source inspection on 2026-09-22, using the existing local raw
intake as cached-verified context. No new raw recount, geometry conversion,
scientific role, targets, model fitting or performance evaluation. This is an
unsuccessful conversion-unblocking attempt, not new independent scene support.

## Evidence and Limits

The original author's TrackNPred repository is available at commit
`3cf1dfca7e98f86e67561b6327c210d84a50cdd4`. Its
[README](https://github.com/rohanchandra30/TrackNPred/blob/3cf1dfca7e98f86e67561b6327c210d84a50cdd4/README.MD)
links to a TraPHic-only repository that currently returns 404. The historical
TRAF dataset page also returns 404 through the current web fetch. This is an
access observation, not proof that no copies exist elsewhere.

The inspected
[box visualizer](https://github.com/rohanchandra30/TrackNPred/blob/3cf1dfca7e98f86e67561b6327c210d84a50cdd4/model/Tracking/drawBB.py)
uses width/height arithmetic, but its paths name an Alibaba dataset and its
row format is not the local TRAF frame/count/agent format. It does not resolve
the local xyxy-versus-xywh conflict. Likewise, the
[hypothesis formatter](https://github.com/rohanchandra30/TrackNPred/blob/3cf1dfca7e98f86e67561b6327c210d84a50cdd4/model/Tracking/hypo_formatter.py)
handles tracker-output columns, not those raw TRAF annotation rows.

The author's
[scale example](https://github.com/rohanchandra30/TrackNPred/blob/3cf1dfca7e98f86e67561b6327c210d84a50cdd4/computer_vision_2d-to_3d/pixel_to_3D.m)
uses hard-coded camera arrays and two manually chosen image points; its comments
describe an approximate scale using a known object dimension and missing depth.
It contains no verified mapping of these constants to our thirty local files.
It is not per-recording homography evidence and was not executed.

In the author's
[issue response](https://github.com/rohanchandra30/TrackNPred/issues/8#issuecomment-536351724),
not all TRAF videos were used in the paper. The question immediately preceding
that answer concerns missing and unexpected class labels. Therefore, reported
paper results cannot validate every local file's types or geometry.

A related coauthor repository was also inspected at
`a03be68326fb8f13ae919f02d85f542ec3997435`:
[Spectral-Trajectory-Prediction](https://github.com/rayguan97/Spectral-Trajectory-Prediction/tree/a03be68326fb8f13ae919f02d85f542ec3997435).
Its data-processing tree provides Argoverse, Apolloscape and Lyft adapters, not a
parser linking the local TRAF boxes to the required convention. No third-party
code was executed or included in the reproduction archive.

## Decision

Keep the existing geometry-conversion refusal and quality quarantine. Do not
infer agent centers, metric scale, a top-down static camera, physical-site
independence or data-use permission from these code examples. The new information
rules out this attempted shortcut; it does not prove the width/height hypothesis
false. A format specification bound to the actual files, a corresponding
annotation exporter, or verified image/annotation correspondence is still needed.

TRAF remains a lower-priority independent pedestrian source than resolving the
already identified DUT/DroneCrowd admission conditions. No pending permission or
scientific-role decision was silently approved. No contacted author, new dataset
download, accepted warning or remote HPC login is claimed.
