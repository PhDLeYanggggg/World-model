# Offline Annotated Observation Decision and Fit-Only Visual Comparison

The user delegated the choice of the more viable scientific route: "哪一步更容易达成目标就走哪一步".
The selected route is standard offline annotated trajectory forecasting. Supplied
history annotations may depend on retrospective interpolation controls. This is
not a strict sensor-as-of, online detection or physically calibrated experiment.
The prior diagnostic receipts remain immutable, including their then-pending status.

The approved parent task, data roles and primary metric remain unchanged: eight
native annotation observations predict twelve; past-normalized ADE, physical-scene
aggregation; raw-frame t+50 remains separate. No metric/seconds/true-3D claim.
ETH ETH, Hotel and grouped Zara01/02/03 are the three fit folds. All 11,966 complete
fit windows remain, including Zara03 with no image. Students development, calibration
and confirmation are unopened. Historical exposure is not erased by this comparison.

Hypothesis: masked past RGB carries cross-scene predictive information beyond
history, neighbors, motion baselines and image-coverage clues. Prespecified arms:
geometry (also receives eight scalar coverage values), mask-only CNN, current RGB,
past RGB. Same geometry preprocessing, model width, zero-initialized CV skip,
minibatches, loss and update budget. No learned camera/Jacobian inputs. Per-pixel
coverage prevents artificial padded pixels being mistaken for measured content.
Current-image latent is repeated to match past-image fusion width.

Use seeds 17/29/43 and leave-one-physical-fit-scene out (36 fits). Fixed 2,000 updates,
batch 64, AdamW, log1p per-row normalized ADE training loss. This loss is declared
separately from the unchanged primary evaluation. No output amplitude cap, threshold
search or held-fold checkpoint selection. Final-update checkpoints only. Strong
motion reference selected on the training fold, not the held scene; also report CV.
Report every arm/seed/site and paired image-minus-geometry differences. Three-site
bootstrap is exploratory and cannot certify deployment or independent replication.

A 100-update timing/recovery pilot resumes the same registered full fit, without
held-fold evaluation at the pilot boundary. Checkpoints, sampler state and heartbeat
are mandatory. Do not stop or shrink data just because CPU training is slow.

ETH/Hotel use the previously inspected supplied-H row/column convention and native
annotation video indices. Physical synchronization and units remain unverified.
Zara01/02 reuse the verified VSP image mapping and explicit partial crops. Source
control provenance is never an input. Future paths appear only in training losses
and evaluation; goal endpoints, legacy teachers and central velocities are absent.
Missing image rows are retained. A visual null result must remain a null result.
Stage5C and SMC remain disabled; this study does not declare submission readiness.
