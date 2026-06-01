# Stage43-BL Raw-Scene / Graph-Rich Gap Matrix

| blocker | status | evidence | required next artifact |
| --- | --- | --- | --- |
| `raw_scene_or_verified_sdf_tensor_missing` | `open` | Stage43 full-waypoint cache has row geometry and future labels, but no raw scene image patch, scene_raster, scene_sdf, walkable_sdf, or homography tensor. | `stage43_raw_scene_patch_or_sdf_cache with train-only construction and row alignment hash` |
| `graph_rich_all_agent_tensor_missing` | `open` | Current model features include scalar neighbor/density/TTC summaries; cache lacks edge_index/edge_attr/all-agent history tensors. | `stage43_all_agent_graph_cache with past/current-only edges, masks, and no future inputs` |
| `scene_goal_contribution_proxy_only` | `partially_supported` | Stage43-AG fresh retrained scene proxy variants show t50/hard signal, but claim boundary says not raw image/SDF. | `retrained raw-scene/SDF ablation: full_raw_scene vs no_scene with bootstrap or multiseed` |
| `interaction_contribution_not_graph_rich` | `partially_supported` | Stage43-AH no_neighbor_interaction shows feature-family contribution, Stage43-X has interaction proxy labels, but neither is graph-rich all-agent dynamics. | `retrained graph-rich ablation: full_graph vs no_graph under same protected policy` |
