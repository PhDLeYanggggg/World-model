# Cross-Recording Intervention Policy

Status: research-only negative candidate; not deployed.

Architecture: 385 input dimensions, 32 hidden SiLU units, one scalar intervention
logit, 12,385 parameters. A frozen original binary decision contributes an offset
of -1 or +1. Inputs combine 380 causal features and five frozen moment predictions.
Original fit-only source normalization and eligibility are retained.

Training: original-TRAIN recording episodes, symmetric one-step second-order
meta-objective, hard-forward sigmoid surrogate, exact unadapted record-risk
dual updates. Train 2,000 updates per fit; 3 head seeds across 6 producer/controller
contexts; same-record matched control. No inference-time adaptation or new
trajectory forecaster. Unknown targets enter supervised completion bounds, not
inference features. Decisions use frozen final logit>0.

Evaluation: 144 fixed fit/inner-held views, paired source bootstrap. Inner-held
was historically exposed and is not independent evaluation or full-pipeline OOF.
All results are conditional on existing source data, producers and moment heads.
Do not interpret repeated windows/contexts/head seeds as independent participants.

Failure: primary 0/72 held budget views; no demonstrated advantage over matched
same-record training. Easy preservation fails. Nonintervention remains available;
an empty conditional-risk denominator is not converted to a passing risk result.
Do not replace any existing deployed selector with these weights.

Coordinates retain the dataset's pixel/local annotation conventions. Prediction
steps are raw annotation units; no metric/time calibration or physical-safety
guarantee follows. This is not true 3D, a foundation model, latent generation,
SMC, independent validation or a CVPR-ready contribution.
