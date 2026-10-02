# Exact-Curvature Solver Amendment

All scientific questions, quadratic objectives, positive mean-preserving form,
data, partitions, normalization, zero-leaf rule, original/additive/Poisson
controls, thresholds,2% risk budget and advance criteria remain those registered
in european_cost_aligned_positive_harm_v1. No independent role is opened.

The real first-head pilot failed its convergence criterion on tree0. Training
only reproduction on the same106,960 known rows failed at128,512 and2048
iterations, maximum gradients1.60e-4,2.63e-6 and1.89e-6. This is a solver failure,
not a measured validation failure of the quadratic hypothesis. No complete
v1 cost model/checkpoint or new validation readout was produced.

The Gauss-Newton approximation discarded residual curvature of the normalized
exponential link. This amendment uses its full analytic Hessian, checked
independently against finite differences. A training-only test of the same
failed tree converged in7 iterations,0.744s, maximum gradient9.97e-8. No
validation model selection was used. The stationary tolerance remains1e-7,
iteration cap128, zero initialization and penalty1 unchanged.

For p=mu*exp(beta*z)/E(exp(beta*z)), define d=z-E_tilt(z).
The Jacobian is p*d. The exact curvature is
`p*(d*d_transpose - Cov_tilt(z))`.
Retain the residual-weighted curvature term in the squared-loss Hessian.
Shift eigenvalues below1e-4 times the ridge penalty for a descent direction;
use TRAIN-only Armijo search, at most32 halvings. Nonconvergence fails explicitly.
This supports convergence, not a global-minimum guarantee for a nonconvex loss.

Reuse the registered data/streaming runner unchanged, replacing only the solver
module and owned output namespace. Preserve the failed implementation, protocol,
registration and real training-only diagnosis. Fresh pilot, then all72 heads
with exact refits, serialization, inference and cached-control checks. Local
native-arm64 compute,0 workers,4 threads,12h cap, checkpoints stored only in the
new owned CREATE M3W directory; no remote scientific computation or new local
numerical cache. Do not touch simulation jobs or unrelated staged work.

The same nominal3000 locality bootstrap and strict advance screen apply. Never
discard unknown labels or replace selected-positive-harm risk with whole-easy
net degradation. No transfer/deployment/confirmation claim from a pilot.
Data remain exposed-development detector-silver, image-local obs8/pred12 at
rawstride12. No metric/seconds/physical-safety/true3D/foundation/submission-ready
claim. Stage5C and SMC stay off.
