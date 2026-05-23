# Stage 22 SDD Data Audit

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold。
- SDD benchmark is pixel-space; no metric claim.
- effective_seconds_unknown; raw-frame horizon only.

- scenes/videos/tracks/rows: `8` / `60` / `10300` / `10616256`
- split videos: `{'train': 40, 'val': 4, 'test': 16}`
- split scenes: `{'train': ['bookstore', 'coupa', 'deathCircle', 'gates', 'hyang'], 'val': ['little'], 'test': ['nexus', 'quad']}`
- same scene across splits: `False`
- annotation frame stride mode: `1`
- raw-frame samples: `{'t+10': 10420614, 't+25': 10266249, 't+50': 10009005, 't+100': 9497463}`
- lazy loading required: `True`
- Stage 22 training allowed: `True`
