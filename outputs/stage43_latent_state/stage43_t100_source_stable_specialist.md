# Stage43-T Source-Stable H100 Specialist

- source: `fresh_stage43_t_t100_source_stable_specialist`
- result_source: `fresh_source_stable_trajnet_crowds_h100_specialist`
- gate: `12 / 12`
- verdict: `stage43_t_source_stable_h100_specialist_deployable`
- deploy source-stable h100 specialist: `True`
- positive h100 dynamics signal: `True`
- validation source safe: `True`
- easy safe: `True`

## Source-Level Split

- train rows: `20284`
- val rows: `6088`
- test rows: `1440`
- train sources: `OpenTraj/datasets/TrajNet/Train/crowds/students001.txt, OpenTraj/datasets/UCY/zara02/obsmat.txt, OpenTraj/datasets/UCY/zara01/obsmat.txt, OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara02.txt, OpenTraj/datasets/UCY/zara03/crowds_zara03.txt`
- val sources: `OpenTraj/datasets/TrajNet/Train/crowds/students003.txt, OpenTraj/datasets/TrajNet/Train/crowds/arxiepiskopi1.txt`
- test sources: `OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt`

## Selected Specialist

- target: `residual`
- l2: `100000.0`
- validation ADE lift: `2.56%`
- validation easy degradation: `0.00%`
- min validation source lift: `2.33%`
- max validation source easy degradation: `0.00%`

## Held-Out H100 Candidate Test Metrics

- rows: `1440`
- full-waypoint ADE improvement: `2.59%`
- endpoint FDE improvement: `-0.55%`
- hard/failure ADE improvement: `2.59%`
- easy degradation: `0.00%`
- bootstrap ADE CI: `[2.06%, 3.14%]`

## Deployment Metrics

- deployment ADE improvement: `2.59%`
- deployment easy degradation: `0.00%`
- deployment reason: `deployable_source_stable_h100_specialist`

## Interpretation

Stage43-T tests whether the only h100 family with enough source coverage, TrajNet_crowds, can support a source-stable long-horizon specialist. Deployment requires validation-source safety and held-out easy preservation; otherwise Stage43-P/R remain the safety floor and t100 remains fallback-only outside diagnostic research.

Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.
