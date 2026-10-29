# Exploratory Objective-Repair Result

We tested whether decoupling cost estimation from action optimization improves
intervention-risk estimates while holding benefit and reference predictions
fixed. Eighteen trained risk heads were continued under two matched objectives,
using identical initial weights, optimizer states, sample streams and2,000
additional updates. Estimation-only continuation reduced source-normalized
all/easy signed-risk MSE across eligible fitting rows by80.29%/55.65% relative
to the frozen original. However, easy-risk MSE on the fixed original selected
cohort did not improve, and only4 of18 context-seed pairs improved over both
controls. Under the unchanged decision rule, estimation-only increased selected
occurrences from62,902 to419,270 and produced known easy-risk violations in62
of72 views. Its preregistered fitting routing screen therefore failed.

These results concern previously exposed training recordings, not independent
generalization. They support the narrower observation that improved global cost
fitting does not by itself establish reliable risk estimates after selection.
They do not establish superiority of action-coupled learning in general, because
both continuations start from action-trained states and the intervention removes
regret and dual penalties together. No candidate was deployed and no new held
evaluation was opened from this experiment. Three risk-head seeds are not three
independent forecaster trainings; repeated windows/views are not independent
observations. All units remain image/dataset-local and raw-frame based.
