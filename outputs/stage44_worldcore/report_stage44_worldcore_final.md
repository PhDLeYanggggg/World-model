# Stage44 WorldCore Final Report

- source: `fresh_stage44_worldcore_latent_state_architecture`
- result_source: `fresh_stage44_worldcore_training_eval`
- verdict: `stage44_worldcore_latent_state_candidate_pass`
- gate: `16 / 16`
- best variant: `hybrid_no_scene`
- protected all/t50/hard: `37.49%` / `20.32%` / `38.82%`
- easy degradation: `1.06%`
- t100 raw diagnostic: `-9.99%`

Stage44 implements a latent world-state architecture rather than another selector-only threshold pass. The current deployable floor is still protected; deployment is not changed by this run.

Ablation read: best variant is `hybrid_no_scene`. In this run, Transformer/SSM contributes, while JEPA, scene proxy, and static interaction tokens are not yet supported as independent main contributions.

Boundary: dataset-local/raw-frame 2.5D only. No metric/seconds claim, no true 3D/foundation claim, no Stage5C, no SMC.
