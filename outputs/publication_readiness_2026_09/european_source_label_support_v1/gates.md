# Label-Support Probe Gates

These are component statuses, not M3W submission-readiness gates.

|Check|Status|Evidence|
|---|---|---|
|Prospective definition|pass|Committed original eighteen-fit design before training|
|Original SOURCE roles|cached_verified|Four fitting/eight excluded localities; no final-role access|
|Causal feature boundary|checked|Geometry/detector input schema rejects future-mask argument; labels separate|
|Complete actual fitting|pass|18models,2304primary trees,128full-refit and128pilot trees|
|Real recovery/full refit|pass|Tree states and held probabilities exact|
|All predictions before scores|pass|18arrays frozen before held-SOURCE reductions|
|Complete inference/scalar replay|pass|18probability arrays and144metric groups exact|
|Registered detector-history effect|positive|AUROC+0.103922 [0.059967,0.162186]|
|Absolute held-SOURCE discrimination|partial|AUROC0.706249; some localities below chance|
|Calibration sufficient for safety|not established|Critical082ECE0.145512; no held calibration selected|
|New robust-bound target constructor|unit checked only|Observed moments unchanged; unknown bounds not fake future labels|
|Bound-aware critic/policy lift|not_run|No critic trained on the new target constructor|
|Complete original policy risk|not_run|A support classifier is not a selective forecasting policy|
|Independent confirmation/submission|false|SOURCE remains exposed development; broader objective unmet|
|Stage5C / SMC|false|No execution|

No recording is excluded, undefined metrics are retained, and no classifier
threshold is chosen from these results. Pixel/raw-frame/released-silver only.
