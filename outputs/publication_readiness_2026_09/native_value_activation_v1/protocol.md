# Continuous Value And Risk For Native Activation

Use the same native 8-observation / 12-prediction task, three actual
forecaster seeds, 34 exposed development recordings, and fitting-only
cuts. Reuse the exact 42 activation folds (10 fit / 3 calibration /
1 scored source), candidate bank, additive gate and feasibility
ranker. A query's chosen option is unchanged; only activation changes.
Do not open DroneCrowd or change the original 2% recording rule.

Fit five fixed boosted regressors per fold: conservative gain,
all/easy positive harm with the original missing-label penalty, and
all/easy CV reference error. Divide targets by the mean past motion
scale of selected agents. Donor slates exclude the donor source as
well as the held and calibration sources from the additive fit.
Weights balance sources and queries; gate-eligible options receive
four times the within-query weight. Candidate labels are used only
for fitting or evaluation. Models are not tuned on the scored source.

Compare expected gain alone and expected gain minus the two positive
risk excesses (each harm minus 2% of its reference), with penalty
coefficient one. This is a score, not a replacement risk rule. The
prior predicted gate remains unchanged. The primary contrast is
risk-adjusted value minus the previous gain score at equal counts,
matched by recording and agent-count bucket. Replaying the whole
recording for that comparison is retrospective, not deployment.
The two actual current-query variants use thresholds from the same
three donor calibration sources, targeting the same coverage as
before, without held labels or later queries.

Report conservative utility, local feasibility, original recording
risk, relative-path ADE and completion bounds, and prediction MAE
on the eligible selected options. Use 3,000 paired source draws after
averaging seeds. Any reduction in coverage or undefined risk remains
visible; it is not counted as an empirical pass.
