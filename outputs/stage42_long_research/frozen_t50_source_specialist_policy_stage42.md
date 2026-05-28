# Frozen Stage42-IM T50 Source-Specialist Policy

- source: `cached_verified_stage42_ik_il_t50_source_specialist_policy_freeze`
- policy_name: `stage42_im_frozen_t50_source_specialist_policy`
- deployment_role: `source_specialist_t50_raw_frame_policy`
- selection_scope: `prevalidated_source_routing_no_new_threshold_selection`
- default_route: `stage42ii_t50_gain_harm_ensemble`
- ucy_route: `stage42x_row_aligned_ucy_full_waypoint_specialist`
- sha256: `902cf924c7581bb0d2b5efb8eced961c98dcb52d36812b7cf3b7d51a3a6317b2`

## Test Summary

- ADE all/t50/t100raw/hard: `15.88%` / `10.45%` / `18.07%` / `16.37%`
- FDE t50: `26.37%`
- easy degradation: `0.00%`

Claim boundary: source-specialist composition evidence only; protected dataset-local/raw-frame 2.5D; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
