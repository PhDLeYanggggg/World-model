# Stage 14 EWAP t+100 Mask Audit

- Stage12 source-level t+100 episodes: `320`
- Stage12 per-agent complete past+target rows: `0`
- Stage12 per-agent last-past+target rows: `0`
- Source-track t+100 windows in EWAP obsmat: `81`

Diagnosis: Stage12 counted source/episode horizon availability, but Stage13 required per-agent causal past plus target mask in all-agent windows; no row satisfied that stricter policy.
