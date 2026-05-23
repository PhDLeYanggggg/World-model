# Stage 23 SDD Medium Data Audit

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold。
- SDD 是 pixel-space official benchmark，不是 metric benchmark。
- effective_seconds_unknown; raw-frame horizon only.
- This run is configured as quick-plus unless the medium command completes explicitly.

- scenes/videos/tracks/rows: `8` / `60` / `10300` / `10616256`
- current split videos: `{'train': 40, 'val': 4, 'test': 16}`
- same scene across current splits: `False`
- annotation frame stride mode: `1`
- lazy loading: `True`
- medium training allowed: `True`
