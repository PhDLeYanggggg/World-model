# Stage 16 Oracle Distillation Report

- number of oracle labels: `595`
- train/val/test labels: `{'test': 122, 'train': 473}`
- t+50 labels: `514`
- t+100 labels: `81`
- correction_needed rate: `0.410084`
- hard/failure coverage: `595`
- residual direction clusters: `{'angle_bin_0': 75, 'angle_bin_4': 113, 'angle_bin_3': 136, 'angle_bin_1': 29, 'angle_bin_2': 70, 'angle_bin_5': 6, 'angle_bin_6': 32, 'angle_bin_7': 134}`
- failure label distribution: `{'unknown': 351, 'long_horizon_drift': 39, 'speed_change': 175, 'wrong_turn': 21, 'density_congestion': 9}`
- learnable structure: `True`

Oracle labels may use future as supervision labels only. They are not inference inputs, and test split oracle labels are evaluation-only.
