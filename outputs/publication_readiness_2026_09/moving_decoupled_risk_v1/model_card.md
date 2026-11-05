# Model And Data Card

Purpose: a fixed intervention-risk learning diagnostic. The frozen trajectory
producer is not retrained. These heads do not generate new trajectories.
Three own-B contexts, three prespecified architectures, one producer seed.
Original492 causal features: past/history/neighbor and frozen-candidate
rollout context with an explicit64-dimensional motion basis. Previous latent
input columns remain zeroed as in the parent experiment; no new JEPA claim.
Complete current-query pooling; no future endpoint/validity mask at inference.
Preprocessing, target and proposal/observed-mask hashes are bound to each
own-fitting checkpoint. Outcomes and their masks are used only for losses
and evaluation. Old source/teacher lineage restrictions are inherited.

Outputs: all/easy signed risk regression and cost-weighted decision logits.
Both signs must accept, plus original support requirements. Neither score
is a certified upper risk bound. Decision magnitudes are not probabilities.
Shared64/shared128/separate64 have32,260/64,516/64,260 parameters.
AdamW .0001, fixed2,000 updates, original query sampler, no V early stopping.
Both heads retain conservative unknown-outcome supervision.

Evidence: all9 real fits and exact initial/final metric replay complete.
Additional held evaluation: not_run after failed fitting screen.
No independent test, deployment upgrade or submission-ready claim.
Native obs8/pred12 image coordinates; no metric/seconds/true3D/foundation
or human-gold status. Original24-group primary remains NA, not passed.
Private checkpoints and caches are not in Git. Public hashes alone do not
make training reproducible without the permitted local data assets.
Stage5C and SMC remain disabled.
