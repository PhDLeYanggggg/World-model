# User Action Required: Stage42-DV Calibration Candidates

优先确认以下数据源的官方 terms/source identity/path/version/time-geometry semantics。确认前不得转换为 official metric/seconds subset。

| priority | dataset | source candidates | required action | blockers |
| ---: | --- | --- | --- | --- |
| `95` | `ucy_crowd_original` | UCY_students03, UCY_zara01, UCY_zara02, UCY_zara03 | confirm UCY official terms/credit requirements, source identity, local path/version, and H.txt convention before conversion | terms/source_identity/path_version_not_confirmed |
| `95` | `eth_biwi_original` | ETH_seq_eth, ETH_seq_hotel | confirm ETH/BIWI terms, source identity, local path/version, annotation fps, and meter-coordinate convention before conversion | terms/source_identity/path_version_not_confirmed |
| `55` | `trajnetplusplus_official` | none | confirm TrajNet++ official terms, train/val/test split license, and ndjson fps/coordinate convention before conversion | terms/source_identity/path_version_not_confirmed, homography_or_coordinate_transform_missing |
| `20` | `stanford_drone_dataset` | none | keep SDD as already converted pixel raw-frame reference; do not count as new external metric/time source | terms/source_identity/path_version_not_confirmed, homography_or_coordinate_transform_missing, fps_or_annotation_timestep_missing, already_sdd_reference_not_external_expansion |
| `10` | `opentraj_toolkit` | none | confirm OpenTraj toolkit role as loader/mirror only; do not treat toolkit as dataset license by itself | terms/source_identity/path_version_not_confirmed, homography_or_coordinate_transform_missing, fps_or_annotation_timestep_missing |
| `10` | `aerialmpt_or_other_topdown` | none | provide official URL, terms, and raw data identity before conversion | terms/source_identity/path_version_not_confirmed, homography_or_coordinate_transform_missing, fps_or_annotation_timestep_missing |
| `0` | `tgsim_diagnostic` | none | diagnostic only; do not use as pedestrian top-down official benchmark | terms/source_identity/path_version_not_confirmed, homography_or_coordinate_transform_missing, fps_or_annotation_timestep_missing, traffic_diagnostic_not_pedestrian_topdown_official |
