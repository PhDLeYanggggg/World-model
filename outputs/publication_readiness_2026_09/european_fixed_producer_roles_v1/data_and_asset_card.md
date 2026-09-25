# Data And Frozen Assets

Source: EuropeanSquares released detector tracks. Coordinate unit: image pixels.
Observation8/prediction12 native annotation steps at raw stride12. This is not
historical SDD t+50, not seconds or metric coordinates. Detector-derived tracks
are not human-gold labels or guaranteed complete visibility of every agent.

The existing twelve opened source localities form three four-locality groups.
Fresh role-index checks found116,823,23,762 and178,384 indexed rows, respectively,
for318,969 unique index entries before their repeated use across rotations. These
are heavily overlapping trajectory windows, not independent observations.

Each ordered rotation uses A for the cached four-source producer, B for new cost
supervision, and C for development readout. Every original source can occupy each
role across rotations; none is thereby an independent confirmation source. Source
hashes and lineage are checked through the frozen parent chain. Model-selection,
calibration and confirmation sources remain unopened.

## Verified Reuse

- Cached data, causal geometry and trajectory forecasts: cached_verified, not newly collected or retrained.
- Fresh9 neural checkpoint-prefix replays use each checkpoint's train-selected baseline, not the CV comparison index.
- Fresh72 old cost-head replays and36 cropped original stopping-policy replays match exactly.
- All original forecast and floor producers exclude C. Matched B forecasts also exclude B from producer fitting. OOF-control B forecasts exclude the scored source within B and all C.
- Original A-derived easy/hard cutoffs apply to both new arms and C; B supplies new feature/error normalization and training labels only.
- Source rosters and role-index hashes are recorded in the private immutable assets_complete.json; its receipt is included in the public completion evidence.

The new controller feature vector has380 dimensions constructed from past
geometry and causal baseline/candidate rollout diagnostics. Future trajectories
provide loss/evaluation targets only. Future-label absence stays unknown; it is
never converted into zero error or an easy label. Predictions can exist for rows
with no future labels, whose intervention counts are disclosed without inventing
accuracy. No new candidate goals, endpoint heatmaps or future-derived inputs.

This experiment does not add full scene images, new agents, raw-video modalities,
trajectory-forecaster training, independent calibration or a new physical dynamics
head. Public aggregates cannot replace access to the private source/checkpoint
chain. Third-party tracks, feature stores, checkpoints and row predictions stay
out of Git.
