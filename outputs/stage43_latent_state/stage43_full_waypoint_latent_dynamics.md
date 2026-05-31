# Stage43-M Protected Full-Waypoint Latent Dynamics

- source: `fresh_stage43_m_full_waypoint_latent_dynamics`
- result_source: `fresh_run`
- mode: `small`
- checkpoint committed: `False`
- gate: `11 / 11`
- verdict: `stage43_m_protected_full_waypoint_latent_candidate_pass`
- deploy neural full-waypoint head: `True`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D multi-agent world-state candidate。
- full waypoints / future endpoints 只作为 loss/eval label，不作为 inference input。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Protected Test Metrics vs Full-Waypoint Floor

- rows: `16000`
- full-waypoint ADE improvement: `29.77%`
- endpoint FDE improvement: `39.59%`
- t50 full-waypoint ADE improvement: `16.45%`
- t50 endpoint FDE improvement: `28.82%`
- t100 raw-frame diagnostic: `-17.79%`
- hard/failure full-waypoint ADE improvement: `28.75%`
- easy degradation: `0.00%`
- switch rate: `68.91%`

## Bootstrap CI

- bootstrap n: `1000`
- full-waypoint ADE improvement CI: `[28.90%, 30.69%]`
- t50 full-waypoint ADE improvement CI: `[14.88%, 17.87%]`
- hard/failure ADE improvement CI: `[27.79%, 29.71%]`
- easy degradation CI: `[0.00%, 0.00%]`

## Ungated Neural Diagnostic

- full-waypoint ADE improvement: `16.29%`
- t50 full-waypoint ADE improvement: `16.25%`
- hard/failure full-waypoint ADE improvement: `18.13%`
- easy degradation: `55.72%`

## No-Leakage Boundary

- future endpoint input: `False`
- future waypoint input: `False`
- future waypoint label/eval only: `True`
- central velocity input: `False`
- test endpoint goal construction: `False`
- test statistics normalization: `False`

## Interpretation

Stage43-M provides a protected full-waypoint latent dynamics candidate under the frozen floor. It is still dataset-local raw-frame 2.5D evidence, not metric/seconds-level or generative world-model execution.
