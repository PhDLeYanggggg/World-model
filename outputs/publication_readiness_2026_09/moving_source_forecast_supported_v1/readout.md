# Admitted Moving-Source Forecast Controls

## Material Passport

Fresh_run: 18 real Torch CPU producer fits, three actual training seeds,
complete excluded-source inference and the two first-job full refits.
Cached_verified: every saved checkpoint/prediction and a separate full
prediction replay. This report recomputes baseline scalars on the exact
same hash-verified in-memory population; writing prose is not training.

## Scope And Primary

**The original 24-group primary is not estimable and has not passed.**
All 64 admitted recordings are processed, but 58 recordings/23 groups
have fixed clip-contained 8/12 queries. Three short scene2 clips give
one wholly empty group. Other zero-query clips remain in the ledger.
The support repair removes no data row and fabricates no zero errors.

Rows: 195,286; known 190,570; complete 136,295;
wholly unknown 4,716; endpoint labels 141,061.
The raw input inventory includes current-visible partial-history agents.
Both comparator interfaces consume the same complete-past nearest8neighbors.
The separate48shared observed-motion features are not used by either model
and have no claimed downstream contribution in this experiment.

## Supported-Group Secondary Results

Paired completion-lower EqMotion-minus-PCT difference: **+14.4599 [-8.1907, +38.9277] pp**.
Known-label paired difference: **+55.0152 [+27.8895, +87.3852] pp**.
Intervals use3,000source-group resamples after the two excluded producers
and three actual seed averages. They are not window-level IID intervals.
The legacy interval field says locality; these are conservative source
components, not certified independent geographic sites. Available-label
ADE on partial rows is not full-future ground truth. Unknown rows use
the unchanged paired-disagreement bound, not imputed positions.

| Model | Equal supported-group pixel ADE | Available-endpoint diagnostic FDE (px) | Mean groupwise gain vs CV (%) |
|---|---:|---:|---:|
| constant_position | 195.6881 | 422.2377 | -323.0727 |
| constant_velocity_causal_fd | 61.3680 | 159.2642 | 0.0000 |
| damped_velocity_090 | 83.2048 | 242.3055 | -57.5618 |
| damped_velocity_097 | 66.6038 | 182.2247 | -14.5612 |
| history_ols4_velocity | 61.9310 | 156.8218 | -2.0686 |
| history_ols8_velocity | 71.4986 | 168.9479 | -20.9970 |
| PCT | 77.3815 | 212.0057 | -39.2724 |
| eqmotion | 53.4925 | 132.7579 | 15.7428 |

Endpoint FDE uses valid endpoint labels only, then averages all supported
groups; it is not full-population FDE. Gains average groupwise ratios,
not the ratio of the two displayed ADE means. Native8/12scores are not
directly comparable with historical SDD raw-frame horizon scores.

Baselines are reported separately. A held-data best baseline is diagnostic,
never an inference choice; each producer chooses its actual rollout floor
using fitting groups only. PCT and EqMotion are architecture-family controls,
not capacity-matched or an isolated equivariance effect. EqMotion emits one
fixed head, not published best-of20metrics.

## Actual Training And Reproduction

Complete18-model train invocation PID78359: 11361.552s, peak RSS 2,149,498,880bytes.
Original excluded-source inference PID91599: 5555.304s.
Full verification/refit invocation PID98674: 6638.724s.
Primary budget is72,000updates;200of them reuse exact first-job pilot prefixes.
Direct-pilot checks and8,000full-refit replay updates are additional verification
work, not extra primary fitting budget. All nine paired families have identical
train IDs,loss factors,draw coverage and sampler RNG. Held rows are never drawn.
All original predictions and both complete first-job checkpoints replay exactly.
CPU4/interop1/workers0; no NumPy model fallback, MPS claim or resource probing.

## What This Does Not Establish

No controller, matched intervention policy, selected-risk calibration, full2%
operational/easy contract, physical consistency or independent confirmation
has passed here. No JEPA/scene/goal/shared-motion lift is demonstrated.
DUT backgrounds and audit future images are not predictive inputs.
Native annotation steps/pixels are not seconds/metres or equal durations
to SDD/European views. This is not true3D,foundation or a new dynamics
architecture contribution. Deployment stays unchanged; Stage5C/SMCoff.
The full research goal and submission-readiness standard remain unmet.
