# OpenDD: learn a residual direction without increasing displacement

The deterministic direction menu increased the exact frontier at
matched displacement. Fit a smooth angular generator with a
two-parameter linear-in-horizon logit and angle bounded to 90 degrees.
Its only inputs are the observed path in the local disagreement frame,
causal scale, and frozen CV-to-OLS4 interpolation fraction. Every
generated trajectory has exactly the original stepwise CV-relative
norm. It may change the endpoint direction. No target coordinates
enter inference.

Compare an observed-error objective against a conservative dual-reference
objective. The latter maximizes the worse paired gain against CV and
OLS4, subtracting each comparison's path disagreement on missing steps.
Both use the same rows at fixed interpolation fractions 0.25, 0.5 and
1, architecture, zero-rotation initialization, minibatches, and 400
updates. Training labels come from the other six locations. No
learned teacher, holdout tuning, or calibration is used.

First keep the prior 101 guarded exchanges and the same actual action
within each view. Also report the separately fixed all-eligible mask
with 296 actions and the prior ranker's fixed forced picks. Every arm
has 780 original edits and unchanged per-step displacement, including
the unchanged recipients and paths. Include no-swap, original, fixed
acceleration, fixed turn and causal suffix choice. Report the original
2% checks, available native errors, conservative paired gain and
3,000 location-bootstrap draws after averaging three seeds.
No new external outcomes or release claim.
