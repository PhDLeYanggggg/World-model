# Causal Cap-Event Probe

## Intended Use
Research diagnosis of transferable risk information for baseline-relative
neural forecasting. Not a trajectory forecaster,physical-risk predictor,
calibrated abstention policy or deployable controller. Models estimate a
producer-relative binary event,not expected error magnitude.

## Models and Inputs
Matched linear logistic and one-hidden-layer SiLU32 heads. Native causal
vector383,context7,context missingness7,risk fractions2:399 inputs total.
Linear head400 parameters;MLP12,833. The causal vector contains observed
history,local interaction summaries and frozen forecast/policy features,
not actual future trajectories. Risk inputs are H/envelope and H_E/H from
fitting-locality-excluded producers. Fitting-only imputation and scaling
follow the registered protocol. Output is one event probability per query.

## Label Semantics
The event is realized positive-CV-easy excess cost above frozen predicted
all-harm. The positive-CV easy threshold is the25th percentile of positive
fitting CV costs. Future labels are used only for supervision/evaluation.
Unknown labels and zero causal disagreement are not sampled for training.
Perfect-CV queries are not members of this positive-CV easy event. They must
retain their separate zero-reference protection rule;low cap-event probability
does not mean no harm on that stratum,or on the remaining population.

Here harm means an increase in forecasting error relative to a reference,
not injury,collision ground truth or physical safety. Targets originate in
released detector trajectories,not human gold. Obs8/pred12 annotation steps,
image pixels only. No metric,time-in-seconds,true3D or foundation claim.

## Training and Provenance
Three fixed seeds,six producer/controller assignments,two input families,
four held source localities per assignment. All288 heads receive2000 updates.
No hyperparameter selection on held outcomes. Inner OOF risk producers fit
two localities;outer producers fit three. This remaining producer transport
is documented rather than treated as identical score distributions.
Model/optimizer/RNG checkpoints and per-row predictions remain private.
Public prediction-freeze receipts bind each completed checkpoint and score
file. They are not training data or weights in the public repository.

## Evidence Boundary
Fixed-batch loss decreases are optimization evidence only. Source-held
classification,ranking and probability scores are evaluated after prediction
freeze;none alone demonstrates trajectory improvement. Four-locality
bootstrap intervals and three seeds do not supply independent confirmation.
Source-development outcomes have prior exposure. Independent selection,
reserved calibration and confirmation are not opened. No deployment change
is allowed from this probe. Stage5C and SMC remain off.
