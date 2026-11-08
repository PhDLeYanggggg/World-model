# Secondary DroneCrowd calibration diagnostics

The original decision is unchanged. This is not independent confirmation.

| Arm | Available ADE (px) | Available step-12 FDE (px) | Easy ADE (px) | Hard ADE (px) |
| --- | ---: | ---: | ---: | ---: |
| constant_velocity_causal_fd | +3.038 [+2.848, +3.228] | +5.546 [+5.203, +5.909] | +3.336 [+3.210, +3.453] | NA |
| history_ols4_velocity | +1.468 [+1.328, +1.637] | +2.647 [+2.318, +3.034] | +1.574 [+1.436, +1.733] | NA |
| independent | +3.025 [+2.834, +3.216] | +5.522 [+5.178, +5.888] | +3.319 [+3.193, +3.439] | NA |
| scene | +3.024 [+2.834, +3.214] | +5.520 [+5.177, +5.885] | +3.318 [+3.193, +3.437] | NA |
| pair_additive | +2.985 [+2.798, +3.172] | +5.447 [+5.112, +5.804] | +3.276 [+3.152, +3.394] | NA |
| joint | +2.984 [+2.798, +3.171] | +5.446 [+5.112, +5.803] | +3.276 [+3.152, +3.393] | NA |
| gate_0 | +3.038 [+2.848, +3.227] | +5.545 [+5.203, +5.908] | +3.336 [+3.210, +3.452] | NA |
| gate_1 | +3.038 [+2.848, +3.227] | +5.545 [+5.203, +5.908] | +3.336 [+3.210, +3.452] | NA |
| gate_2 | +3.038 [+2.848, +3.227] | +5.545 [+5.203, +5.908] | +3.336 [+3.210, +3.452] | NA |
| gate_3 | +3.038 [+2.848, +3.227] | +5.545 [+5.203, +5.908] | +3.336 [+3.210, +3.452] | NA |

| Joint vs same-count control | Observed pair gain (px) | Full-grid lower (px) | Full-grid upper (px) |
| --- | ---: | ---: | ---: |
| independent | +0.087 [+0.070, +0.104] | +0.082 [+0.067, +0.099] | +0.089 [+0.072, +0.107] |
| scene | +0.085 [+0.070, +0.102] | +0.081 [+0.066, +0.096] | +0.088 [+0.072, +0.104] |
| pair_additive | +0.001 [+0.001, +0.002] | +0.001 [+0.001, +0.001] | +0.002 [+0.001, +0.002] |

Positive pair gain favors joint. Missing steps are bounded by forecast disagreement, not treated as zero error.
Intervals are paired source bootstraps over 15 groups after seed averaging (3,000 draws).
NA denotes missing source-seed support. Units are image pixels and annotation steps.
