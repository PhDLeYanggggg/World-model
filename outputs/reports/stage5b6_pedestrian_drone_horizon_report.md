# Stage 5B.6 Pedestrian / Drone Long-Horizon Repair

This report distinguishes actual converted local data from registry-only or license/manual placeholders. Downsampled horizons are not counted as original verified t+100 unless explicitly available.

| dataset_name | coordinate_unit | metric_or_pixel | max_raw_horizon | t50_verified | t100_verified | t50_samples | t100_samples | download_status | suitable_for_official_gate | why_or_why_not |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stanford Drone Dataset | unknown | unknown | 0 | False | False | 0 | 0 | downloadable | False | not downloaded; non-commercial/license/manual preparation remains unresolved |
| TrajNet++ | dataset_coordinate | pixel_or_dataset_coordinate | 10 | False | False | 0 | 0 | downloadable | False | local converted source only supports short horizon |
| ETH/UCY | dataset_coordinate | pixel_or_dataset_coordinate | 10 | False | False | 0 | 0 | not_in_registry_or_not_downloaded | False | local converted source only supports short horizon |
| OpenTraj-compatible pedestrian datasets | unknown | unknown | 0 | False | False | 0 | 0 | not_in_registry_or_not_downloaded | False | registry/planning entry only; no actual converted local episodes |
| AerialMPT longer sequences | pixel_or_unknown | pixel_or_dataset_coordinate | 12 | False | False | 0 | 0 | not_in_registry_or_not_downloaded | False | local converted source only supports short horizon |

Conclusion: no real pedestrian/drone source with verified t+50/t+100 was added in this run. TGSIM remains the only verified long-horizon source family, and it is traffic/generic trajectory data.