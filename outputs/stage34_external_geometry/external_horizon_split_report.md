# Stage34 External Horizon/Split Report

- source: `fresh_run`
- split strategy: `file-level split from Stage31; scene-level held-out diagnostic where test scene not in train`
- held-out external scenes: `['biwi']`
- same scene across splits: `True`
- splits: `{'train': {'rows': 119109, 'scene_count': 2, 'file_count': 6, 'horizon_counts': {10: 38978, 25: 33700, 50: 28743, 100: 17688}, 'frame_step_counts': {10: 38978, 30: 33165, 50: 28743, 100: 17688, 26: 535}}, 'val': {'rows': 7685, 'scene_count': 1, 'file_count': 1, 'horizon_counts': {10: 2465, 25: 2175, 50: 1885, 100: 1160}, 'frame_step_counts': {10: 2465, 30: 2175, 50: 1885, 100: 1160}}, 'test': {'rows': 3636, 'scene_count': 2, 'file_count': 4, 'horizon_counts': {10: 2020, 25: 1212, 50: 404}, 'frame_step_counts': {10: 2020, 30: 1212, 50: 404}}}`
- horizon status: `{'t10': 'available_dataset_local_raw_frame', 't25': 'available_dataset_local_raw_frame', 't50': 'available_dataset_local_raw_frame_but_small_test_rows', 't100': 'train_val_available_but_test_unavailable_diagnostic_only'}`
