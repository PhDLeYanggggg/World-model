# Past observation quality and selective forecast utility

The previous fixed-candidate ranker transferred poorly. Keep its
candidate, quarter amplitude and two changes per query. Test new
signals from the same eight observed steps, not extra history:
released head box size/shape, jitter, image-edge clearance,
OLS4-predicted edge clearance, overlap and past visibility of
nearby current tracks. Use first-frame image dimensions already
verified for the same annotation hash. Target histories are
complete by roster design; do not invent a target gap feature.
Box features are observation proxies, not detector confidence,
physical occlusion truth or verified online tracking.

Four matched 500-step Torch fits per fold/seed: direct gain and
factorized gain, each with and without box-quality features.
Use identical draws and model size; zero the feature block in
motion-only arms. All arms train a per-step support auxiliary.
Factorization uses expected gain
`p_t * mu_t * d_t - (1-p_t) * d_t`, where `d_t` is the known
causal candidate disagreement, `p_t` is probability of an
available label, and `mu_t` is bounded relative improvement
conditional on availability. True future masks/coordinates are
training and scoring targets only. A previous support model
predicted wholly unlabeled rows; this test resolves individual
steps and adds new raw-box evidence.

Three seeds and nine held fitting sources. Freeze all scores
before comparing. Report label-support Brier/AUROC/AUPRC against
fold-local prevalence, and separately test full-grid decision
utility, ADE/FDE, easy/hard and same-nomination regret. Main
paired contrasts are factorized quality vs factorized motion,
direct quality vs direct motion, and factorized quality vs
OLS4 and minimum change, all retaining their appropriate counts.
Bootstrap 3,000 source groups with replacement, averaging seeds
within source first. Higher support accuracy is not a gain
unless the matched policy results also improve. Preserve all
original fit rows and bound every missing step.

No extra data role, external labels, calibration or confirmation;
no policy threshold selection. Keep the original 2% rule.
No deployment, Stage5C, SMC, metric or seconds claim.
