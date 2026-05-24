# Stage35 External Data Expansion Report

- source: `fresh_run`; local OpenTraj/ETH/UCY files are `cached_verified` inputs.
- candidate files: `18`
- converted files: `18`
- splits: `{'train': {'rows': 158942, 'scenes': 6, 'scene_ids': ['ETH_UCY_seq_eth', 'ETH_UCY_seq_hotel', 'ETH_UCY_students03', 'ETH_UCY_zara01', 'ETH_UCY_zara02', 'TrajNet_biwi'], 'agents': 435, 'horizon_counts': {10: 46604, 25: 42907, 50: 38943, 100: 30488}, 'track_length_median': 38.0}, 'val': {'rows': 112746, 'scenes': 2, 'scene_ids': ['TrajNet_crowds', 'TrajNet_mot'], 'agents': 892, 'horizon_counts': {10: 37683, 25: 32059, 50: 26756, 100: 16248}, 'track_length_median': 20.0}, 'test': {'rows': 66303, 'scenes': 3, 'scene_ids': ['UCY_crowds', 'UCY_students01', 'UCY_zara03'], 'agents': 892, 'horizon_counts': {10: 21267, 25: 18765, 50: 16263, 100: 10008}, 'track_length_median': 20.0}}`
- coordinate unit: `dataset_local_coordinates_mixed_sources`
- metric status: `unverified_weak_metric_diagnostic; homography files may exist but not calibrated into common metric frame`
- image/homography availability: `{'ETH_UCY_has_reference_or_H_files': True, 'TrajNet_has_scene_images': False}`
- held-out scene split supported: `True`
- unsupported/empty files: `[]`
