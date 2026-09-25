# Model Card: Coherent Event Cost Head

This small neural head predicts cost moments, not new trajectories. It cannot
establish a new world-dynamics contribution simply by learning to select among
fixed forecasts. No deployment is changed by completing this source-only study.

Architecture: 383 causal features, width64 GELU layer, four raw outputs.
The decoded moments are D_all, H_all, D_easy, H_easy, with nonnegative costs,
H_all bounded by the causal rollout envelope and easy costs nested within all
costs. Future easy labels are used as supervision, not inference membership.

Both arms minimize B-only component-RMS-normalized squared cost error. The
selected arm additionally penalizes group-mean residuals in each training
locality crossed with population, frozen utility proposals and frozen raw
policy selections. It has no extra parameters. Old neural utility remains
fixed. Neither loss is claimed to implement a multicalibration theorem.

Seventy-two fits cover two forecast pairs, two objectives, six ordered source
assignments and three seeds. Every fit has the same 2,000-update budget;
paired arms share initialization, draws, normalizers and supported labels.
Atomic checkpoints include optimizer/RNG/draws/trace for exact resume. No
selection-data early stopping, threshold fitting or test-driven best-seed choice.

Individual all/dual gates, uniform-query switching and greedy-query allocation
are evaluated separately. The greedy policy spends predicted all/easy harm
budgets within a same-frame query; it is neither optimal nor collision-aware.
A query-count-matched hash control is diagnostic and does not match risk.
Predicted budget compliance never substitutes for measured harm constraints.

Limitations include in-sample B teacher selection, small and previously opened
C rosters, future-label missingness, detector quality, population shift and
imperfect moment prediction. No independent risk certificate, physical safety,
metric/seconds, true3D, foundation or human-gold claim. Stage5C and SMC remain off.
