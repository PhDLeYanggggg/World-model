# Stage42-BO User Action Required: Calibrated Subset

- source: `fresh_calibrated_subset_source_cv`

## Before Stronger Metric / Seconds Claims

- Manually verify homography direction and coordinate convention for each calibrated source in the fold table.
- Confirm whether each horizon is an annotation-step horizon at 2.5fps / 0.4s, not a raw video-frame horizon.
- Keep global M3W reports as raw-frame / dataset-local unless evaluation is explicitly restricted to these calibrated sources.
- Do not use this as a true-3D, foundation, Stage5C, or SMC claim.
