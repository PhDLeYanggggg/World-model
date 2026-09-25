# Causal Abstention Results

Fresh causal guards and readout; cached_verified forecasters, heads and population.
No neural training in this experiment. All 720 registered views evaluated, without selecting a deployment.

Ranges below span 36 dependent development views per policy. Positive/negative CI counts
are descriptive, not independent hypothesis tests or multiplicity-corrected evidence.

| Parent / policy | All-ADE gain vs floor (%) | Gain vs original (%) | Positive / negative CI vs original | Easy worst locality gain vs CV (%) | Zero-CV harm views |
|---|---:|---:|---:|---:|---:|
| cv_targets__original | 0.1249 to 1.7632 | 0 to 0 | 0 / 0 | -0.3001 | 24 / 24 supported |
| cv_targets__stop | 0.1249 to 1.7634 | -2.89221e-06 to 0.000116997 | 0 / 0 | -0.3001 | 0 / 24 supported |
| cv_targets__stop_risk | 0.1249 to 1.7634 | -2.89221e-06 to 0.000116997 | 0 / 0 | -0.3001 | 0 / 24 supported |
| cv_targets__stop_random | 0.1249 to 1.7634 | -2.89221e-06 to 0.000116997 | 0 / 0 | -0.3001 | 0 / 24 supported |
| cv_targets__support | 0.0883 to 1.6472 | -0.1278 to -0.0181 | 0 / 36 | -0.4369 | 0 / 24 supported |
| cv_targets__support_risk | 0.0883 to 1.6976 | -0.1392 to -0.0188 | 0 / 36 | -0.4369 | 0 / 24 supported |
| cv_targets__support_random | 0.0883 to 1.6909 | -0.1046 to -0.0180 | 0 / 36 | -0.4369 | 0 / 24 supported |
| cv_targets__combined | 0.0883 to 1.6472 | -0.1278 to -0.0181 | 0 / 36 | -0.4369 | 0 / 24 supported |
| cv_targets__combined_risk | 0.0883 to 1.6976 | -0.1392 to -0.0188 | 0 / 36 | -0.4369 | 0 / 24 supported |
| cv_targets__combined_random | 0.0883 to 1.6909 | -0.1046 to -0.0180 | 0 / 36 | -0.4369 | 0 / 24 supported |
| floor_both__original | 0.1225 to 1.6409 | 0 to 0 | 0 / 0 | 0.0777 | 24 / 24 supported |
| floor_both__stop | 0.1225 to 1.6410 | -3.83662e-06 to 0.000118551 | 0 / 0 | 0.0777 | 0 / 24 supported |
| floor_both__stop_risk | 0.1225 to 1.6410 | -3.83662e-06 to 0.000118551 | 0 / 0 | 0.0777 | 0 / 24 supported |
| floor_both__stop_random | 0.1225 to 1.6410 | -3.83662e-06 to 0.000118551 | 0 / 0 | 0.0777 | 0 / 24 supported |
| floor_both__support | 0.0880 to 1.5241 | -0.1224 to -0.0149 | 0 / 33 | -0.0509 | 0 / 24 supported |
| floor_both__support_risk | 0.0881 to 1.5633 | -0.1078 to -0.0146 | 0 / 33 | -0.0509 | 0 / 24 supported |
| floor_both__support_random | 0.0880 to 1.5457 | -0.1040 to -0.0139 | 0 / 33 | 0.0192 | 0 / 24 supported |
| floor_both__combined | 0.0880 to 1.5241 | -0.1224 to -0.0149 | 0 / 33 | -0.0509 | 0 / 24 supported |
| floor_both__combined_risk | 0.0881 to 1.5633 | -0.1078 to -0.0146 | 0 / 33 | -0.0509 | 0 / 24 supported |
| floor_both__combined_random | 0.0880 to 1.5457 | -0.1040 to -0.0139 | 0 / 33 | 0.0192 | 0 / 24 supported |

## Same-Frame Count-Matched Evidence

| Policy | Control | All gain range (%) | Positive / negative CI | Complete-label gain range (%) | Hard gain range (%) |
|---|---|---:|---:|---:|---:|
| cv_targets__stop | risk | 0 to 0 | 0 / 0 | 0 to 0 | 0 to 0 |
| cv_targets__stop | random | 0 to 0 | 0 / 0 | 0 to 0 | 0 to 0 |
| cv_targets__support | risk | -0.0513 to 0.0524 | 8 / 6 | -0.0589 to 0.0931 | -0.0912 to 0.0635 |
| cv_targets__support | random | -0.0481 to 0.0041 | 5 / 5 | -0.0435 to 0.0118 | -0.0769 to 0.0043 |
| cv_targets__combined | risk | -0.0513 to 0.0524 | 8 / 6 | -0.0589 to 0.0931 | -0.0912 to 0.0635 |
| cv_targets__combined | random | -0.0481 to 0.0041 | 5 / 5 | -0.0435 to 0.0118 | -0.0769 to 0.0043 |
| floor_both__stop | risk | 0 to 0 | 0 / 0 | 0 to 0 | 0 to 0 |
| floor_both__stop | random | 0 to 0 | 0 / 0 | 0 to 0 | 0 to 0 |
| floor_both__support | risk | -0.0457 to 0.0397 | 6 / 4 | -0.0474 to 0.0484 | -0.0778 to 0.0318 |
| floor_both__support | random | -0.0473 to 0.0032 | 4 / 4 | -0.0457 to 0.0036 | -0.0793 to 0.0004 |
| floor_both__combined | risk | -0.0457 to 0.0397 | 6 / 4 | -0.0474 to 0.0484 | -0.0778 to 0.0318 |
| floor_both__combined | random | -0.0473 to 0.0032 | 4 / 4 | -0.0457 to 0.0036 | -0.0793 to 0.0004 |

## Lost Benefit and Avoided Harm

Both quantities are divided by the same locality-specific floor-error sum, then averaged over the fixed roster.
Their difference is a percentage-point change in gain over the floor, not a relative percentage vs the parent.

| Policy | Avoided all harm (pp) | Lost all benefit (pp) | Net all change (pp) | Net hard change (pp) | Switch-rate range |
|---|---:|---:|---:|---:|---:|
| cv_targets__original | 0 to 0 | 0 to 0 | 0 to 0 | 0 to 0 | 0.0050 to 0.4119 |
| cv_targets__stop | 0 to 0.000136592 | 0 to 2.13192e-05 | -2.84341e-06 to 0.000115273 | -1.0264e-06 to 0.000110838 | 0.0050 to 0.4114 |
| cv_targets__stop_risk | 0 to 0.000136592 | 0 to 2.13192e-05 | -2.84341e-06 to 0.000115273 | -1.0264e-06 to 0.000110838 | 0.0050 to 0.4114 |
| cv_targets__stop_random | 0 to 0.000136592 | 0 to 2.13192e-05 | -2.84341e-06 to 0.000115273 | -1.0264e-06 to 0.000110838 | 0.0050 to 0.4114 |
| cv_targets__support | 0.0030 to 0.0256 | 0.0224 to 0.1435 | -0.1254 to -0.0181 | -0.1537 to 0.0020 | 0.0045 to 0.3889 |
| cv_targets__support_risk | 0.0027 to 0.0542 | 0.0273 to 0.1873 | -0.1367 to -0.0187 | -0.1365 to 0.0053 | 0.0045 to 0.3889 |
| cv_targets__support_random | 0.0024 to 0.0331 | 0.0250 to 0.1238 | -0.1027 to -0.0179 | -0.0963 to 0.0001 | 0.0045 to 0.3889 |
| cv_targets__combined | 0.0030 to 0.0256 | 0.0224 to 0.1435 | -0.1254 to -0.0181 | -0.1537 to 0.0020 | 0.0045 to 0.3889 |
| cv_targets__combined_risk | 0.0027 to 0.0542 | 0.0273 to 0.1873 | -0.1367 to -0.0187 | -0.1365 to 0.0053 | 0.0045 to 0.3889 |
| cv_targets__combined_random | 0.0024 to 0.0331 | 0.0250 to 0.1238 | -0.1027 to -0.0179 | -0.0963 to 0.0001 | 0.0045 to 0.3889 |
| floor_both__original | 0 to 0 | 0 to 0 | 0 to 0 | 0 to 0 | 0.0042 to 0.3282 |
| floor_both__stop | 0 to 0.000136595 | 0 to 2.02107e-05 | -3.78611e-06 to 0.000116868 | -2.41937e-06 to 0.000110838 | 0.0042 to 0.3279 |
| floor_both__stop_risk | 0 to 0.000136595 | 0 to 2.02107e-05 | -3.78611e-06 to 0.000116868 | -2.41937e-06 to 0.000110838 | 0.0042 to 0.3279 |
| floor_both__stop_random | 0 to 0.000136595 | 0 to 2.02107e-05 | -3.78611e-06 to 0.000116868 | -2.41937e-06 to 0.000110838 | 0.0042 to 0.3279 |
| floor_both__support | 0.0034 to 0.0727 | 0.0202 to 0.1364 | -0.1202 to -0.0148 | -0.1420 to 0.0092 | 0.0037 to 0.3170 |
| floor_both__support_risk | 0.0033 to 0.0735 | 0.0241 to 0.1364 | -0.1067 to -0.0146 | -0.0977 to 0.0132 | 0.0037 to 0.3170 |
| floor_both__support_random | 0.0031 to 0.0728 | 0.0224 to 0.1212 | -0.1020 to -0.0138 | -0.1138 to 0.0130 | 0.0037 to 0.3170 |
| floor_both__combined | 0.0034 to 0.0727 | 0.0202 to 0.1364 | -0.1202 to -0.0148 | -0.1420 to 0.0092 | 0.0037 to 0.3170 |
| floor_both__combined_risk | 0.0033 to 0.0735 | 0.0241 to 0.1364 | -0.1067 to -0.0146 | -0.0977 to 0.0132 | 0.0037 to 0.3170 |
| floor_both__combined_random | 0.0031 to 0.0728 | 0.0224 to 0.1212 | -0.1020 to -0.0138 | -0.1138 to 0.0130 | 0.0037 to 0.3170 |

Complete per-mode, fold, seed, event, subset and locality results are in the adjacent JSON/CSV files.
CSV includes p95/p99 and worst-locality errors; unknown outcomes are never scored as zero.
Query cohorts are the indexed agent histories at a current frame, not all visible agents.
The comparison is not an interaction-conflict metric or a physical-safety certificate.
