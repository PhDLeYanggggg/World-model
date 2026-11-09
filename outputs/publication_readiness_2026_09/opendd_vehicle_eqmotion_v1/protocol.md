# A task-matched EqMotion vehicle forecasting control

Use the existing seven OpenDD vehicle location folds, unchanged 8/12
native grid, same cohort, CV, OLS4 and bounded completion-trained vehicle
incumbent. Do not reuse the earlier pedestrian-focused EqMotion weights.
Fit the pinned CVPR 2023 EqMotion author core as a deterministic K=1
adaptation, head zero fixed before fitting. No best-of-20 comparison or
reproduction claim.

Fit two variants with the same architecture, sampling, optimizer, update
budget and seeds. In the focal comoving frame, one reads three nearest
observed neighbors; the other masks every neighbor. The aligned origins
were reconstructed from the same past-only OpenDD rows. At train and
inference, cap correction from CV at the incumbent's same two-unit
bound in observed-scale coordinates. Train using its same normalized
completion objective and exact donor sampling, 400 updates, batch 128,
seeds 17/29/43. Each fit excludes the held site. Use the fixed final
checkpoint, with no held model selection. Four arm64 CPU compute
threads, zero data workers, optimizer/sampling state every 100 updates.

Freeze all 42 checkpoints before held scoring. Replay the incumbent;
report available-step and conservative paired utility vs CV, OLS4 and
incumbent. Compare peers directly with ego on the same labels, and
report native ADE/FDE, tail and original easy/risk summaries. Use the
unchanged 3,000 location draws after averaging seeds. This is an
OpenDD development comparison of forecast producers, not a release
policy or independent confirmation. No Songdo or DroneCrowd outcomes,
no new scientific split, metric unit or risk tolerance.
