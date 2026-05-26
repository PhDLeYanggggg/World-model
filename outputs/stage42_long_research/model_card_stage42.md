# Stage42 Model Card

## Model

M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under Stage37/teacher floor.

## Intended Use

Research evaluation for dataset-local raw-frame top-down multi-agent trajectory/world-state prediction and failure/hard-case diagnostics.

## Not Intended For

- metric or seconds-level physical deployment
- true 3D world modeling
- large-scale foundation model claims
- autonomous deployment without dataset/domain validation
- Stage5C latent generative execution or SMC

## Performance Summary

- protected external all: `0.2103`
- protected external t50: `0.1365`
- protected external t100 raw-frame diagnostic: `0.1469`
- protected external hard/failure: `0.2038`
- protected easy degradation: `0.0000`
- full-waypoint ADE all/t50: `0.1858` / `0.1480`

## Safety

The Stage37/teacher floor is required for current deployment. Ungated neural is explicitly rejected.

<!-- STAGE42_AC_REFRESH:START -->
## Stage42-AC Latest Evidence Refresh

- source: `fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- Stage42-AB is now included as auxiliary-head evidence.
- Auxiliary-head conclusion: mixed / partial; not a uniform main contribution claim.
- Stage5C and SMC remain disabled.

### Model-Card Update

Interaction/occupancy/physical heads are present in the full-waypoint model interface, but Stage42-AB shows they should be described as auxiliary diagnostics/regularizers with mixed evidence, not as uniformly beneficial deployable heads.
<!-- STAGE42_AC_REFRESH:END -->
