# Stage 20 Conversion Report

- Registry-only data is not counted as converted.
- Download failures and user-action-required sources are not counted as converted.
- Light raw-index conversion is not the same as full benchmarked world-state episodes.

- successful light conversions/indexes: `5`

| dataset | conversion level | official eval candidate | actual t+50 | actual t+100 | path |
| --- | --- | --- | --- | --- | --- |
| OpenTraj supported datasets | raw_index_plus_sample_audit | True | False | False | /Users/yangyue/Downloads/World/external_data/OpenTraj |
| UCY Crowd original | raw_index_plus_sample_audit | True | False | False | /Users/yangyue/Downloads/World/data/stage5b_raw/trajnetplusplusdataset/data/trajnet_original/crowds |
| full ETH/UCY / EWAP | archive_index_only | True | False | False | /Users/yangyue/Downloads/World/data/stage5b_raw/trajnetplusplusdataset/data/ewap_dataset_light.tgz |
| Stanford Drone Dataset | raw_index_plus_sample_audit | True | False | False | /Users/yangyue/Downloads/World/external_data/StanfordDroneDataset |
| TrajNet++ full datasets | raw_index_plus_sample_audit | True | False | False | /Users/yangyue/Downloads/World/data/stage5b_raw/trajnetplusplusdataset |
