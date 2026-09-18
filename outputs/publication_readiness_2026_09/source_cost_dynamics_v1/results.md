# Source Trajectory Cost Alignment Results

Fixed source-fit diagnostic, not the main benchmark or independent confirmation.
Parent-normalized ADE averaged within physical site, then equally across sites. Average seed losses, not ensemble paths.
The fixed0.9 gate is not calibrated safety. All sites and seeds are retained.

| Objective | Input | Mode | ADE | CV ADE | Gain vs CV (%) [conditional95% site interval] | Positive sites / fits | Mean site easy harm | Mean site intervention |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| ade | mask_only | uncontrolled | 1033.937290 | 1017.000770 | -1.665340 [-2.367498, -1.229130] | 0/5; 0/15 | 23.464676 | 1.0000 |
| ade | mask_only | fixed_probability_gate | 1017.050678 | 1017.000770 | -0.004907 [-0.008067, -0.002006] | 0/5; 0/15 | 0.062920 | 0.0026 |
| ade | past_rgb | uncontrolled | 1034.272967 | 1017.000770 | -1.698347 [-2.598328, -1.171436] | 0/5; 0/15 | 23.920100 | 1.0000 |
| ade | past_rgb | fixed_probability_gate | 1017.174881 | 1017.000770 | -0.017120 [-0.034629, -0.005228] | 0/5; 0/15 | 0.223939 | 0.0095 |
| log_ade | mask_only | uncontrolled | 1033.342266 | 1017.000770 | -1.606832 [-2.379782, -1.113321] | 0/5; 0/15 | 22.144692 | 1.0000 |
| log_ade | mask_only | fixed_probability_gate | 1017.042814 | 1017.000770 | -0.004134 [-0.006959, -0.001833] | 0/5; 0/15 | 0.055651 | 0.0026 |
| log_ade | past_rgb | uncontrolled | 1032.954584 | 1017.000770 | -1.568712 [-2.335907, -1.028130] | 0/5; 0/15 | 21.730562 | 1.0000 |
| log_ade | past_rgb | fixed_probability_gate | 1017.158485 | 1017.000770 | -0.015508 [-0.031691, -0.004497] | 0/5; 0/15 | 0.198198 | 0.0095 |

## Matched Contrasts

| Comparison | Input/objective | Mode | Gain over matched control (%) [conditional95% site interval] |
| --- | --- | --- | --- |
| RGB_vs_mask | ade | uncontrolled | -0.032466 [-0.170630, +0.059879] |
| RGB_vs_mask | log_ade | uncontrolled | +0.037517 [-0.170461, +0.210902] |
| ADE_vs_log_ADE | mask_only | uncontrolled | -0.057582 [-0.159752, +0.033465] |
| ADE_vs_log_ADE | mask_only | fixed_probability_gate | -0.000773 [-0.001515, -0.000173] |
| ADE_vs_log_ADE | past_rgb | uncontrolled | -0.127632 [-0.282875, +0.029865] |
| ADE_vs_log_ADE | past_rgb | fixed_probability_gate | -0.001612 [-0.004586, +0.000450] |

Gated RGB/mask comparisons do not isolate pixels because their frozen classifier gates differ.
Same-arm loss comparisons share exactly the same gate. No gate, model or checkpoint is selected from these scores.
Easy CV error is exactly zero: relative degradation is undefined, not automatically a pass. See absolute harms and native-pixel errors.
Context bounds are not a physical validity certificate. Closed-ball and binary oracles use future labels for diagnosis only.
Source +144rawframes is not seconds-equivalent to main native8-to12. Pixel coordinates are not metric.
Five exposed source sites and overlapping training folds yield conditional sensitivity only. No deployment, Stage5C or SMC.
