# Stage 23 SDD No-Leakage Report

- split_leakage_by_video: `False`
- split_leakage_by_scene_for_cross_scene: `False`
- same_agent_id_across_split: `agent ids are video-local; episode ids include scene/video/split_type`
- endpoint_leakage_in_goal_construction: `False`
- candidate_goals_train_only: `True`
- velocity_causal_fd_only: `True`
- central_velocity_official: `False`
- future_endpoint_input: `False`
- test_statistics_normalization: `False`
- test_endpoint_heatmap_in_scene_pack: `False`
- goalbench_within_scene_leakage: `False`
- passed: `True`
