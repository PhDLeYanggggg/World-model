# Stage30 Project World Model Gap

- M3W-LAS v2 is stronger than Stage26 on SDD pixel-space raw-frame and has fresh seed/bootstrap support.
- It is still largely a selector + latent-feature policy, not a full generative or metric 3D world model.
- External non-SDD conversion is diagnostic; full M3W-LAS transfer remains blocked by latent/scale alignment.
- Next shortest path: build external non-SDD latent cache and calibrated feature store, then run transfer evaluation without test tuning.
