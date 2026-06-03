# Stage43-CW T100 Residual Admissibility Source Stress

- source: `fresh_stage43_cw_t100_residual_admissibility_source_stress`
- result_source: `fresh_t100_source_scene_single_exclusion_stress`
- verdict: `stage43_cw_t100_source_stress_survives_single_exclusion_diagnostic`
- gate: `10 / 10`
- stress verdict: `source_scene_stress_survives_single_exclusion`
- deploy on current heldout t100: `False`

## Aggregate

- all replay exact: `True`
- all single-source exclusions positive: `True`
- all single-scene exclusions positive: `True`
- min without-source t100 mean: `0.000812`
- min without-scene t100 mean: `0.000812`
- negative source slices mean: `1.00`
- negative scene slices mean: `1.00`

### Worst source removals from first seed

| removed label | slice rows | slice t100 | without-label t100 | removal flips | slice negative |
| --- | ---: | ---: | ---: | --- | --- |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt` | `2638` | `0.001451` | `0.000619` | `False` | `False` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/ETH/seq_eth/obsmat.txt` | `463` | `0.001232` | `0.000777` | `False` | `False` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara02.txt` | `506` | `0.001934` | `0.000804` | `False` | `False` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/zara03/crowds_zara03.txt` | `245` | `0.001630` | `0.000820` | `False` | `False` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/biwi/biwi_hotel.txt` | `203` | `0.002536` | `0.000824` | `False` | `False` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt` | `242` | `0.001185` | `0.000832` | `False` | `False` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/arxiepiskopi1.txt` | `76` | `0.000000` | `0.000846` | `False` | `False` |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/students003.txt` | `963` | `0.000740` | `0.000855` | `False` | `False` |

### Worst scene removals from first seed

| removed label | slice rows | slice t100 | without-label t100 | removal flips | slice negative |
| --- | ---: | ---: | ---: | --- | --- |
| `ETH_UCY_students03` | `2638` | `0.001451` | `0.000619` | `False` | `False` |
| `ETH_UCY_seq_eth` | `463` | `0.001232` | `0.000777` | `False` | `False` |
| `UCY_zara03` | `245` | `0.001630` | `0.000820` | `False` | `False` |
| `TrajNet_biwi` | `203` | `0.002536` | `0.000824` | `False` | `False` |
| `UCY_crowds` | `242` | `0.001185` | `0.000832` | `False` | `False` |
| `TrajNet_crowds` | `2764` | `0.000848` | `0.000838` | `False` | `False` |
| `ETH_UCY_seq_hotel` | `509` | `-0.000151` | `0.000868` | `False` | `True` |
| `ETH_UCY_zara01` | `448` | `-0.000538` | `0.000876` | `False` | `True` |

### Domain removals from first seed

| removed label | slice rows | slice t100 | without-label t100 | removal flips | slice negative |
| --- | ---: | ---: | ---: | --- | --- |
| `TrajNet` | `2967` | `0.000901` | `0.000815` | `False` | `False` |
| `ETH_UCY` | `5333` | `0.000845` | `0.000836` | `False` | `False` |
| `UCY` | `1700` | `0.000724` | `0.000866` | `False` | `False` |

## Interpretation

- This is a source/scene stress audit for the tiny CU/CV t100 residual-admissibility signal.
- Passing this audit does not deploy t100; it only says the supported-protocol signal is not destroyed by removing one source or scene at a time.
- Future endpoints/full waypoints remain labels only; inference inputs are causal.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
