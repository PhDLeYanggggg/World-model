# User Action Required: Stage42-DT Raw Source Parseability

本步骤只证明若干本地源有 trajectory-like / calibration-like 文件形态。继续转换前仍需用户确认 legal/source 信息。

| dataset | dry-run status | required action |
| --- | --- | --- |
| `ucy_crowd_original` | parseable | after user terms/source confirmation, run no-leakage conversion plus source-specific time/geometry audit；并提供 terms/source confirmation |
| `eth_biwi_original` | parseable | after user terms/source confirmation, run no-leakage conversion plus source-specific time/geometry audit；并提供 terms/source confirmation |
| `trajnetplusplus_official` | parseable | after user terms/source confirmation, run no-leakage conversion as dataset-local raw-frame source；并提供 terms/source confirmation |
| `opentraj_toolkit` | not parseable in sample | locate raw trajectory files or official extraction instructions after terms confirmation；并提供 terms/source confirmation |
| `aerialmpt_or_other_topdown` | not parseable in sample | locate raw trajectory files or official extraction instructions after terms confirmation；并提供 terms/source confirmation |
| `stanford_drone_dataset` | parseable | already converted SDD reference; use only for SDD pixel raw-frame work；并提供 terms/source confirmation |
| `tgsim_diagnostic` | not parseable in sample | diagnostic traffic source only; not pedestrian topdown official；并提供 terms/source confirmation |
