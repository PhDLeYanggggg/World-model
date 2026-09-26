# Auxiliary-Cost Source-Development Data

## Provenance and Roles
The study reuses the frozen European top-down detector-trajectory source pool
and its recorded training/producer lineage from the verified cap-event study.
This is cached_verified data, not a newly collected dataset. New feature/target
alignment checks, model fits and metrics are labeled fresh_run separately.
The source-development pool has prior outcome exposure. Independent selection,
reserved calibration and confirmation are not opened, relabeled or reused.

## Schema and Support
Each view has three fitting localities and one source-held locality. Frozen
forecast producers exclude the risk-model localities. The two causal risk
fractions for fitting rows come from inner teachers that exclude that row's
locality; the held row uses its frozen outer teacher. Input schema is399
dimensions, including explicit context missingness. IDs and target hashes
bind row alignment but are not feature embeddings.

The main forecasting protocol is8 observed and12 predicted native annotation
steps. Risk targets are forecast-cost labels, not future trajectory inputs.
Known positive-disagreement rows train the heads. Exact-zero disagreement
cost is structurally zero; missing future labels are not converted into
observed zero-cost examples. Positive-CV easy is defined by the frozen
fitting-only cut. Perfect-CV cases retain their separate guard.

Fresh support checking completed all144 views. Minimum event rows are132
for full and15 for motion-only. Some individual motion localities have no
events. Counts are overlapping windows, not independent people/scenes or a
power calculation. The support report also contains recording-agent and
recording counts. Unsupported readout metrics must remain not_estimable.

## Units and Quality
Detector-image pixels and annotation steps only. This study does not verify
homography, metric scale or elapsed physical time. Detector-derived and
inferred labels are not human gold. No physical collision/safety ground truth
is supplied by an increase in forecast error.

## Storage
Raw data, feature rows, checkpoints, per-row scores, source targets and logs
remain in ignored local directories. Public files contain code, configuration,
aggregate metrics and hashed lineage receipts. No training data or large cache
is included in the public commit. No simulation result substitutes for real
source observations. Stage5C and SMC are not executed.
