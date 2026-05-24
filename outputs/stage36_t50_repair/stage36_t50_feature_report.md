# Stage36 t+50 Feature Report

- source: `fresh_run`; Stage35 rows/labels are `cached_verified`.
- feature dim: `29`
- split report: `{'train': {'rows': 158942, 't50_rows': 38943, 'feature_dim': 29, 'goal_available_fraction': 1.0}, 'val': {'rows': 112746, 't50_rows': 26756, 'feature_dim': 29, 'goal_available_fraction': 0.0}, 'test': {'rows': 66303, 't50_rows': 16263, 'feature_dim': 29, 'goal_available_fraction': 0.0}}`
- no leakage: `{'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'baseline_relative_error_h50': 'label_only', 'oracle_margin_h50': 'label_or_analysis_only', 'track_remaining_length': 'audit_only_excluded_from_inference_features'}`
- `future_endpoint_x/y` are used only through baseline labels/evaluation arrays, never as inference features.
