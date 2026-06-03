# Stage43-CV T100 Residual Admissibility Slice Attribution

- source: `fresh_stage43_cv_t100_residual_admissibility_slice_attribution`
- result_source: `fresh_t100_residual_admissibility_slice_attribution`
- verdict: `stage43_cv_t100_slice_attribution_broad_supported_diagnostic`
- gate: `10 / 10`
- scope verdict: `broad_enough_to_expand`
- deploy on current heldout t100: `False`

## Replay

- all replay exact: `True`
- max replay diff: `0.00000000`

## Seed Summary

| seed | t100 improvement | switch rate | positive sources | max source gain share | max scene gain share | replay diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `4323` | `0.000841` | `0.067000` | `12` | `0.372020` | `0.372020` | `0.00000000` |
| `4331` | `0.001180` | `0.061600` | `12` | `0.527920` | `0.527920` | `0.00000000` |
| `4337` | `0.001501` | `0.070000` | `12` | `0.415717` | `0.415717` | `0.00000000` |

## Concentration

- mean max source positive-gain share: `0.438552`
- mean max scene positive-gain share: `0.438552`
- mean positive sources: `12.00`
- any seed source narrow: `False`
- any seed scene narrow: `False`

## Top Slices From First Seed

### Domain

| label | rows | switched | switch rate | slice improvement | gain share | harm switched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | `5333` | `405` | `0.0759` | `0.000845` | `0.6960` | `1.097924` |
| `TrajNet` | `2967` | `177` | `0.0597` | `0.000901` | `0.2071` | `0.123245` |
| `UCY` | `1700` | `88` | `0.0518` | `0.000724` | `0.0968` | `0.056721` |

### Source file

| label | rows | switched | switch rate | slice improvement | gain share | harm switched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt` | `2638` | `190` | `0.0720` | `0.001451` | `0.3720` | `0.368013` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/ETH/seq_eth/obsmat.txt` | `463` | `104` | `0.2246` | `0.001232` | `0.1927` | `0.231459` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/students003.txt` | `963` | `67` | `0.0696` | `0.000740` | `0.0690` | `0.036292` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara02.txt` | `506` | `48` | `0.0949` | `0.001934` | `0.0624` | `0.064103` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/students001.txt` | `1219` | `57` | `0.0468` | `0.000718` | `0.0615` | `0.022850` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/zara01/obsmat.txt` | `448` | `36` | `0.0804` | `-0.000538` | `0.0558` | `0.159330` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students01/students001-trajnet.txt` | `1213` | `55` | `0.0453` | `0.000454` | `0.0490` | `0.039183` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/ETH/seq_hotel/obsmat.txt` | `509` | `29` | `0.0570` | `-0.000151` | `0.0386` | `0.102610` |

### Scene

| label | rows | switched | switch rate | slice improvement | gain share | harm switched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY_students03` | `2638` | `190` | `0.0720` | `0.001451` | `0.3720` | `0.368013` |
| `TrajNet_crowds` | `2764` | `172` | `0.0622` | `0.000848` | `0.1930` | `0.123245` |
| `ETH_UCY_seq_eth` | `463` | `104` | `0.2246` | `0.001232` | `0.1927` | `0.231459` |
| `ETH_UCY_zara01` | `448` | `36` | `0.0804` | `-0.000538` | `0.0558` | `0.159330` |
| `UCY_students01` | `1213` | `55` | `0.0453` | `0.000454` | `0.0490` | `0.039183` |
| `ETH_UCY_seq_hotel` | `509` | `29` | `0.0570` | `-0.000151` | `0.0386` | `0.102610` |
| `ETH_UCY_zara02` | `1275` | `46` | `0.0361` | `-0.001508` | `0.0369` | `0.236513` |
| `UCY_zara03` | `245` | `19` | `0.0776` | `0.001630` | `0.0275` | `0.008568` |

### Source-agent

| label | rows | switched | switch rate | slice improvement | gain share | harm switched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt|248` | `108` | `22` | `0.2037` | `0.011821` | `0.0736` | `0.018672` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt|340` | `87` | `25` | `0.2874` | `0.005020` | `0.0654` | `0.066335` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt|418` | `91` | `16` | `0.1758` | `0.008197` | `0.0437` | `0.015886` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt|331` | `53` | `12` | `0.2264` | `0.004823` | `0.0313` | `0.020838` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt|324` | `64` | `10` | `0.1562` | `0.004315` | `0.0263` | `0.010499` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt|75` | `73` | `6` | `0.0822` | `0.004826` | `0.0240` | `0.000000` |

## Interpretation

- This step attributes the tiny CU-confirmed t100 admissibility lift across domain/source/scene/source-agent slices.
- If concentration is narrow, the correct next step is slice-specific expansion or heldout stress testing, not a deployment claim.
- Future endpoints/full waypoints remain labels only; inference inputs are causal.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
