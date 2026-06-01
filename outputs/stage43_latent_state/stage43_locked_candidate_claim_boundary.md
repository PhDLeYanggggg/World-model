# Stage43 Locked Candidate Claim Boundary

## Allowed

- protected dataset-local/raw-frame 2.5D multi-agent world-state candidate
- multimodal latent-state heads are useful as protected proxy heads
- latest protected tail-horizon candidate improves all/t50/hard while preserving easy cases
- safety floor remains part of the method

## Not Allowed

- true 3D world model
- large-scale foundation world model
- metric or seconds-level prediction
- ungated standalone deployment
- uniform positive external transfer across every source
- Stage5C execution
- SMC execution

## Practical Wording

M3W currently has protected multimodal latent-state evidence under a safety floor. It is best described as a dataset-local/raw-frame 2.5D multi-agent world-state candidate, not as a true-3D, metric, seconds-level, foundation, or ungated generative world model.
