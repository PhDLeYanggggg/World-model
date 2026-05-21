# Stage 5B.5 Data Card

Actual long-horizon pedestrian/drone data is still missing in this quick run. TGSIM provides verified t+100, but it is traffic/generic trajectory data, not proof of pedestrian world modeling.

## Horizon Audit
| dataset | raw_t50 | raw_t100 | t100_samples | coordinate_unit |
| --- | --- | --- | --- | --- |
| eth_ucy | False | False | 0 | dataset_coordinate |
| tgsim | True | True | 48 | meter |
| tgsim_i90 | True | True | 131 | meter |
| trajnet | False | False | 0 | dataset_coordinate |

## Hard Subsets
| dataset | hard | medium | easy | t100_hard | eval_ok |
| --- | --- | --- | --- | --- | --- |
| eth_ucy | 2 | 11 | 10 | 0 | False |
| tgsim | 3 | 15 | 14 | 3 | True |
| tgsim_i90 | 8 | 9 | 14 | 8 | True |
| trajnet | 6 | 12 | 14 | 0 | True |

SDD remains a license/manual placeholder. It was not downloaded or counted as converted.
