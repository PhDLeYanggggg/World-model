# Stage 5B Data Card

Actual converted datasets are separated from registry-only or gated datasets. Official benchmark inputs use causal finite-difference velocity.

## Converted Datasets

| dataset | domain | coordinate_unit | metric | samples_t100 | actual_verified_t100 | train/val/test |
| --- | --- | --- | --- | --- | --- | --- |
| eth_ucy | pedestrian | dataset_coordinate | False | 0 | False | 15/2/6 |
| tgsim | traffic | meter | True | 483 | True | 23/5/4 |
| tgsim_i90 | traffic | meter | True | 31 | True | 18/7/6 |
| trajnet | pedestrian | dataset_coordinate | True | 0 | False | 19/6/7 |

## Download / Access Records

| dataset | status | kind | executed | notes |
| --- | --- | --- | --- | --- |
| trajnet | planned | git | False | Public GitHub repository with TrajNet++ original trajectory subsets. |
| eth_ucy | planned | derived_from_trajnet | False | Stage 5B uses the BIWI/ETH-style subset bundled in the TrajNet++ repository when present. |
| tgsim_other | planned | url | False | Public Socrata CSV sample used as an additional TGSIM corridor benchmark. |
| sdd | planned | gated_placeholder | False | Stanford Drone Dataset requires license-aware manual preparation; not downloaded by default. |
| opendd | planned | gated_placeholder | False | OpenDD access and license must be verified by the user before local conversion. |
| ngsim | planned | manual_placeholder | False | NGSIM source files must be supplied by the user or an official allowed portal. |


TrajNet++ was cloned from its public GitHub repository. The ETH/UCY fallback in this run is the BIWI/ETH-style file bundled in that TrajNet++ original-data tree, not a separate full official ETH/UCY conversion.
