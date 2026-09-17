# Zara Geometry Repaired; Observation Semantics Remain Open

## Completed Work

`fresh_run`: traced every canonical Zara01/02 source row to its supplied VSP
control points, compared all registered matrix/axis/index alternatives, then
implemented and validated a source-origin adapter. Decoded and inspected six
private contact sheets for 24 deterministic first-history agents, 192 past/current
image requests. This is source repair and input inspection, not model training.

| Source | Stored rows / agents | Complete past-8 windows | Complete 8+12 labels | Coordinate repair |
| --- | ---: | ---: | ---: | --- |
| Zara01 | 5,024 / 148 | 3,988 | 2,234 | H.txt, image row/column, stored frame minus one |
| Zara02 | 9,537 / 204 | 8,110 | 5,741 | Same convention plus a fixed source-origin translation |

Zara02 has 204 stored/raw identities, of which 202 have a complete past-eight
window. These are two recordings at **one physical scene**, not two independent
generalization sites. The 7,975 complete 8+12 windows are source availability,
not an admitted new training cohort or independent sample count. No exactly
stationary eight-step past occurs in these recordings; they do not directly add
support to the earlier stationary-start subset.

## Implemented Repair

The initial audit found no directly matching Zara02 supplied matrix. That failure
is retained in `../zara_media_lineage/audit.json`. An adaptive follow-up fixes the
first canonical source coordinate as a single origin anchor. It is an exact raw
control at source frame 6, not a future forecasting target. The translation is
`[-1.297826081999995, -15.653087400399992]` in the stored native coordinates.
No all-row regression, fitted homography, estimated scale or time-offset search
is used. Every remaining row tests the resulting mapping.

- Zara01 maximum native replay error: 0.00000501199; inverse image error 0.000231750 pixels.
- Zara02 maximum native replay error: 0.00000539712; inverse image error 0.000249526 pixels.
- Maximum consecutive native-displacement mismatch: 0.00000971246 / 0.00000937451.

The supplied matrix plus coordinate convention now numerically reconstructs
the annotations within printed precision. This **does not verify physical scale,
homography calibration accuracy, capture timing or real-time track availability**.
The frame-minus-one relation is established against the VSP source, not an
independent camera clock. Video headers and decoder both report 25; seconds-level
performance remains unclaimed.

## Past Images and Missingness

The first complete history for the first 12 eligible IDs was selected per video,
without checking future survival or future movement. Zara01 has 89/96 valid
96-pixel centered crops and 8/12 complete windows. Zara02 has 70/96 valid crops
and 0/12 complete windows: early observations near image edges are counted as
missing support, not deleted. All 80 distinct requested indexed frames decoded.

All six contact sheets were inspected locally. Visible patches generally contain
nearby pedestrians, but markers do not identify a consistent head/foot/body
location and occlusion is visible. This is qualitative plausibility, not manual
gold localization or verification of every identity. A future input loader must
represent partial crops explicitly rather than silently discarding entering
agents or requiring every crop to be complete. No new forecast or body labels
were inferred from this inspection. Images are not uploaded to Git.

## Important Causality Boundary

The stored coordinates are linearly interpolated from sparse manual controls.
For Zara01, 3,877/3,988 complete past windows (97.22%) require a control after the
query. For Zara02 it is 7,924/8,110 (97.71%). Only 111 and 186 windows respectively
have all contributing control timestamps at or before the query. These counts
did not filter or replace any existing experiment.

Three distinct statements must not be conflated:

1. The model's input API does not read explicit future trajectory targets.
2. A standard offline annotation benchmark can supply stored past positions.
3. A strict online sensor-as-of claim additionally requires those observations
   to have been constructible without later controls.

The new evidence limits statement 3. It is not, by itself, proof of train/test
mixing, nor a reason to erase conventional offline benchmark results. Even the
111/186 mechanically supported windows do not certify online human annotation,
identity or gaze availability. Source VSP controls are dataset-supplied labels;
our inspection is not newly collected human gold.

An asynchronous scientific decision is pending: retain the standard offline
annotation task with an explicit limitation, or rebuild stricter source-as-of
observations under a new protocol. The existing eight-observed/twelve-predicted
parent protocol and normalized primary metric remain unchanged. The separate
prospective primary-metric decision also remains unanswered. No development,
calibration or confirmation labels were opened in this follow-up.

## Verification and Next Action

Registrations: `configs/m3w_zara_media_lineage.json` and
`configs/m3w_zara_past_media.json`. Ten focused tests passed (0.13 seconds); the
unrelated legacy full suite was not rerun. A separate replay reproduced all
numeric source/media fields and byte-identical private manifests/contact sheets.
The original completed outputs were not overwritten. Both processes exited;
there is no live training job from this work.

Next: after the observation definition is confirmed, register a broader moving
past-context comparison within the already approved fit roles, with masked
partial images and identical trajectory-only controls. Do not repeat failed
camera/threshold sweeps on 31 stationary agents, rename Zara videos as new sites,
or use this audit to claim new predictive performance. Broader independent
confirmation support remains a separate unresolved requirement.

Current result: useful input repair, **not yet a supported new prediction method
or submission-ready advantage**. New neural training, deployment, Stage5C and
SMC: not run / disabled.
