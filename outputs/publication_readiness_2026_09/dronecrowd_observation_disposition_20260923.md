# DroneCrowd Observation Disposition

## Delegated Decision, Not a Forecast Result

2026-09-23. Under the author's delegated acquisition/audit/protocol authorization,
the appropriate initial task is **offline trajectory forecasting from exported
historical annotations**. This resolves the previous author-decision dependency
without asserting that missing source-time evidence has been recovered. Model
training and predictive evaluation have not been run by this decision.

## Fixed Source and Input Semantics

- Use the original XML head-box centers as the single coordinate reference.
  Do not silently mix the supplied MAT/text coordinates where the source audit
  found differences. This is image-space head motion, not ground contact points.
- Image `imgSSSFFF.jpg` has XML frame `FFF - 1`. Keep recording-local agent IDs
  namespaced by source and clip; these are not identities across clips.
- Inputs use current and past exported annotation rows only. Keep all currently
  visible agents in the input inventory. Past gaps are masked, not interpolated;
  velocities use backward differences and are invalid across any raw gap.
- Future positions and future visibility remain in a separate supervision path.
  Future-complete trajectories must not determine observed neighbors, scene
  tokens, features or candidate goals. Any learned prototype uses fit roles only.
- XML supplies no reviewed keyframe/generated/interpolated flags. Therefore
  strict sensor-time causality is **unverified**, even when the exported feature
  reader is prefix-invariant. Do not present the offline task as online sensing.

## Horizons and Geometry

Preserve the main eight-observed/twelve-predicted step convention and supplemental
raw t+50. DroneCrowd stride-1 annotation steps are a dataset-local protocol, not
physical-time equivalents of SDD's existing stride-12 development protocol.
Specify both strides in comparisons; do not pool their error numbers as if
sampling periods were equal. FPS/effective seconds remain unknown.

Source-audit registrations estimate image-to-image transforms only. They do not
verify ground-plane homography, metre scale or metric trajectories. Any future
camera stabilization used as an inference input must be estimated from the
available past, not fitted with future frames or the whole test sequence. Until
then image-space motion includes camera motion; report that limitation.

## Roles and Evidence

The delegated authorization permits evidence-based role decisions without another
routine author review. It does not itself establish independent sites. Complete
the cross-clip scene/camera audit first, conservatively group matching or ambiguous
recordings, then freeze the role manifest before model-error readout. Source-audit
images and structural labels are not predictive validation outcomes.

Keep training, selection, calibration and confirmation groups disjoint through
the entire producer chain. The supplied validation folder is sampled from test
and is excluded. Existing exposed SDD scenes remain development data. Automatic
image-match components are only candidate groups, not a certified independent N.

The 2% empirical easy-error condition is unchanged. Do not turn it into a finite-
sample safety claim using a different bounded loss. With too few independent
units, report the bound's insufficiency and baseline fallback rather than relaxing
the tolerance to pass. No Stage5C execution, SMC, new deployment, true-3D,
foundation-model or submission-readiness claim is authorized by this disposition.
