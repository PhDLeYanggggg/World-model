# Stage 23 SDD Dual-Split Report

- cross_scene split: strict scene generalization.
- within_scene_video split: scene/goal learning with train-video endpoints only.
- Do not mix split metrics.

- cross_scene train/val/test scenes: `['bookstore', 'coupa', 'deathCircle', 'gates', 'hyang']` / `['little']` / `['nexus', 'quad']`
- cross_scene video counts: `{'train': 40, 'val': 4, 'test': 16}`
- within_scene video counts: `{'train': 34, 'val': 12, 'test': 14}`
- within_scene videos per scene: `{'bookstore': {'train': ['video0', 'video1', 'video2', 'video3'], 'val': ['video4'], 'test': ['video5', 'video6']}, 'coupa': {'train': ['video0', 'video1'], 'val': ['video2'], 'test': ['video3']}, 'deathCircle': {'train': ['video0', 'video1', 'video2'], 'val': ['video3'], 'test': ['video4']}, 'gates': {'train': ['video0', 'video1', 'video2', 'video3', 'video4'], 'val': ['video5', 'video6'], 'test': ['video7', 'video8']}, 'hyang': {'train': ['video0', 'video1', 'video2', 'video3', 'video4', 'video5', 'video6', 'video7', 'video8'], 'val': ['video9', 'video10', 'video11'], 'test': ['video12', 'video13', 'video14']}, 'little': {'train': ['video0', 'video1'], 'val': ['video2'], 'test': ['video3']}, 'nexus': {'train': ['video0', 'video1', 'video2', 'video3', 'video4', 'video5', 'video6'], 'val': ['video7', 'video8'], 'test': ['video9', 'video10', 'video11']}, 'quad': {'train': ['video0', 'video1'], 'val': ['video2'], 'test': ['video3']}}`
- leakage audit risk: `low if split_type is respected; metrics must not be mixed`
