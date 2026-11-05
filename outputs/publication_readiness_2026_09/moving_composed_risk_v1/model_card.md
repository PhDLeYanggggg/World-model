# Composed Decision Model Card

Purpose: cost-aware intervention-risk learning. The trajectory producer is frozen.
Six fixed models: two loss arms in three own-B contexts, one producer seed.
Architecture: separate64 risk and decision encoders,64,260 parameters each model.
492 causal inputs include history, neighbors and frozen candidate rollouts.
As in the parent, previous latent columns are zeroed; no new JEPA contribution.
Current-query pooling uses no future endpoint or label-validity feature.
Own-fitting preprocessing, input/target/proposal/observed-mask hashes are bound.
Future outcomes and validity are supervision/evaluation only.

Outputs: all/easy signed risk estimates and cost-sensitive decision logits.
Both signs must accept and original support must hold. Neither is a calibrated
risk guarantee. The new max-based surrogate trains their per-channel conjunction.
Original risk MSE, decision loss and conservative unknown supervision remain.
Fixed AdamW .0001, clip5,2,000 updates; no validation checkpoint/threshold search.
Fresh independent controls match preceding models exactly. Both encoders update.

Additional held evaluation: fresh_run_exposed_development.
No independent confirmation, deployment upgrade or submission-ready claim.
Native obs8/pred12 image coordinates, not metric or seconds. Not true3D,
foundation or human-gold evidence. Stage5C and SMC remain disabled.
Private checkpoints/caches are not in Git. Permitted local assets are needed
to replay; hashes alone do not constitute a self-contained public benchmark.
