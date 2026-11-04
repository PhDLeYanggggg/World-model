# Reference-head model/data card

Prediction component only. Direct ExtraTrees regression uses64trees; a two-part
comparison uses32event-classification and32conditional-regression trees. All
depth6/minleaf64/maxfeatures32, bootstrapFalse, fit_threads1. Learned node counts
may differ despite equal tree budgets. No novel architecture is claimed.

382inputs: raw local380 descriptors, logcurrentquerysize, logcausalhistoryscale.
Predicted rollouts are producer-excluded; no learned critic output, global
Cscaler, source ID, future truth/mask/easy label or endpoint-derived test goals.
Inner source3fold fitting excludes entire validation sources for every learned
component. No new permanent official roles. Source/known-query weights are
conditioned, not rebalanced, for the positive-reference amount head.

Same195286past-eligible SOURCE rows from64admitted recordings. Predictions cover
the full role-specific roster; unknown futures are not negative event targets.
The estimand is ER>0 and available-label normalized ER, conditional on supplied
supervision; partial12step labels are not completed. It is not physical future
support or an independent record-event probability. No MAR/agent-independence
assumption justifies transporting it to unknown outcomes or finite-record risk.

Source23supported groups are exposed development, not verified IID physical
sites or independent confirmation. Original24primary remains not_estimable.
Reserved20Moving groups, DUT calibration and DroneCrowd confirmation stay closed.
No image/simulation admission, human-gold promotion, metric/homography/seconds,
true3D or foundation claim. Source-time provenance remains limited.

Nativearm64CPU4/interop1/workers0 with sequential forests, checkpoint every8trees,
PID/event heartbeat, complete source/feature/target/weight bindings.9real resume
cells and3full first-job refits;216model bindings/108OOF arrays and108Harrays
replay. No new decision is made, so this study cannot repair trajectory or
original2%deployment gates by itself. Stage5C/SMC off.
