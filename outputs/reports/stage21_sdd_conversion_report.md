# Stage 21 SDD Conversion Report

- No model training was run.
- Conversion uses causal finite differences only.
- Coordinate status remains pixel-space; no metric claim without homography/scale.
- Raw data and large derived shards are ignored by git.

- annotation files: `60`
- scenes/videos: `8` / `60`
- world-state rows: `10616256`
- tracks: `10300`
- samples t+10/t+25/t+50/t+100 raw-frame: `10420614` / `10266249` / `10009005` / `9497463`
- scene split: `{'bookstore': 'train', 'coupa': 'train', 'deathCircle': 'train', 'gates': 'train', 'hyang': 'train', 'little': 'val', 'nexus': 'test', 'quad': 'test'}`
- labels: `{'Biker': 0, 'Pedestrian': 1, 'Skater': 2, 'Cart': 3, 'Car': 4, 'Bus': 5}`
