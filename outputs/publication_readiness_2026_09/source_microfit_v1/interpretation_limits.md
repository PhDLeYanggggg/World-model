# Pre-Fit Interpretation Limits

The audited microcohorts have16 nonzero rows and32 mixed rows, with distinct
scoped training tracks. Their observed-context radii range from111809 to1757119
parent-normalized units, while the full training-complement CV ADE scale is
1127.65. A potentially large local output-Jacobian mismatch is therefore real,
but its role in failed fitting remains a hypothesis before the paired fits.

The decoder comparison changes a variable per-row scaling law, not only a
global learning rate. Both arms have the same per-example output bound and the
same zero initial prediction, but they need not define identical conditional
function families because the neural feature vector does not explicitly carry
the absolute context radius. Any success supports this tested decoder change;
it does not identify gradient clipping as the sole cause or exclude an
inductive-bias effect. The original full60-model result remains unchanged.

Training-cost scale is a global parameter fitted from training labels. The
microcohort selection also uses training labels to isolate feasible nonzero and
zero targets. Neither operation is a future input at inference. Neither may be
carried into held-set filtering or used to claim benchmark improvement.

No main or held-source forecasting, independent validation, risk calibration,
deployment, Stage5C orSMC is performed by this diagnostic.
