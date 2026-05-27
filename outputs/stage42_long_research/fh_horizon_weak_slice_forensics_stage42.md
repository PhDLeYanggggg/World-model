# Stage42-FL FH Weak-Horizon Forensics

- source: `fresh_stage42_fh_horizon_weak_slice_forensics`
- generated_at_utc: `2026-05-27T08:14:59.487957+00:00`
- gate: `15 / 15`
- verdict: `stage42_fl_horizon_weak_slice_forensics_pass`
- analyzed weak horizons: `['TrajNet|100', 'UCY|50', 'UCY|100']`
- root cause counts: `{'oracle_label_low_margin_ambiguous': 3}`
- next action: `train_horizon_specific_row_level_switch_model_with_stronger_history_neighbor_goal_features`

## Slice Findings

## `TrajNet|100`

- rows: `5608`
- diagnostic oracle improvement vs floor: `19.34%`
- diagnostic oracle improvement vs FH: `1.06%`
- low-margin share: `{'0.01': 0.9880527817403709, '0.025': 0.9909058487874465, '0.05': 0.9917974322396577}`
- oracle candidate distribution: `{'fh': 3963, 'fc': 5, 'di': 1502, 'fa': 90, 'fb': 0, 'floor': 48}`
- root causes: `['oracle_label_low_margin_ambiguous']`

### Validation candidate metrics

| candidate | rows | all vs floor | delta vs FH | t50 | t100raw | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fh` | 1160 | `22.64%` | `0.00%` | `0.00%` | `22.64%` | `22.64%` | `1.53%` | `25.00%` |
| `fc` | 1160 | `22.63%` | `-0.01%` | `0.00%` | `22.63%` | `22.63%` | `1.53%` | `25.00%` |
| `di` | 1160 | `23.26%` | `0.80%` | `0.00%` | `23.26%` | `23.26%` | `1.71%` | `25.00%` |
| `fa` | 1160 | `23.26%` | `0.80%` | `0.00%` | `23.26%` | `23.26%` | `1.71%` | `25.00%` |
| `fb` | 1160 | `23.26%` | `0.80%` | `0.00%` | `23.26%` | `23.26%` | `1.71%` | `25.00%` |
| `floor` | 1160 | `0.00%` | `-29.27%` | `0.00%` | `0.00%` | `0.00%` | `-0.00%` | `0.00%` |

### Test candidate metrics

| candidate | rows | all vs floor | delta vs FH | t50 | t100raw | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fh` | 5608 | `18.47%` | `0.00%` | `0.00%` | `18.47%` | `18.47%` | `3.08%` | `31.46%` |
| `fc` | 5608 | `18.49%` | `0.02%` | `0.00%` | `18.49%` | `18.49%` | `3.08%` | `31.47%` |
| `di` | 5608 | `19.01%` | `0.66%` | `0.00%` | `19.01%` | `19.01%` | `3.34%` | `31.60%` |
| `fa` | 5608 | `19.03%` | `0.69%` | `0.00%` | `19.03%` | `19.03%` | `3.34%` | `31.60%` |
| `fb` | 5608 | `19.01%` | `0.66%` | `0.00%` | `19.01%` | `19.01%` | `3.34%` | `31.60%` |
| `floor` | 5608 | `0.00%` | `-22.66%` | `0.00%` | `0.00%` | `0.00%` | `-0.00%` | `0.00%` |

### Validation past-only proxy signal

| candidate | positive gain rate | mean gain vs FH | corr endpoint-floor | corr endpoint-FH | corr path length | corr min distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fc` | `0.09%` | `-0.000133` | `-0.06493779866424922` | `-0.82266572090242` | `-0.03547433445397568` | `0.036255766845869696` |
| `di` | `23.88%` | `0.007265` | `0.439884894802013` | `0.30448021902717753` | `0.31220937237471236` | `-0.03882093819900333` |
| `fa` | `23.97%` | `0.007256` | `0.43774142776602437` | `0.29566132538774664` | `0.3102251416272569` | `-0.037813379098790484` |
| `fb` | `23.88%` | `0.007268` | `0.4400996394074298` | `0.304549636155545` | `0.31232018504272757` | `-0.03888281485741463` |

## `UCY|50`

- rows: `2340`
- diagnostic oracle improvement vs floor: `32.52%`
- diagnostic oracle improvement vs FH: `6.75%`
- low-margin share: `{'0.01': 0.9106837606837607, '0.025': 0.917094017094017, '0.05': 0.9252136752136753}`
- oracle candidate distribution: `{'fh': 1806, 'fc': 29, 'di': 296, 'fa': 24, 'fb': 0, 'floor': 185}`
- root causes: `['oracle_label_low_margin_ambiguous']`

### Validation candidate metrics

| candidate | rows | all vs floor | delta vs FH | t50 | t100raw | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fh` | 2340 | `27.63%` | `0.00%` | `27.63%` | `0.00%` | `27.63%` | `0.04%` | `64.70%` |
| `fc` | 2340 | `27.78%` | `0.20%` | `27.78%` | `0.00%` | `27.78%` | `0.41%` | `65.00%` |
| `di` | 2340 | `22.72%` | `-6.79%` | `22.72%` | `0.00%` | `22.72%` | `-16.34%` | `65.00%` |
| `fa` | 2340 | `22.53%` | `-7.06%` | `22.53%` | `0.00%` | `22.53%` | `-16.22%` | `65.00%` |
| `fb` | 2340 | `22.59%` | `-6.97%` | `22.59%` | `0.00%` | `22.59%` | `-16.27%` | `65.00%` |
| `floor` | 2340 | `0.00%` | `-38.19%` | `0.00%` | `0.00%` | `0.00%` | `-0.00%` | `0.00%` |

### Test candidate metrics

| candidate | rows | all vs floor | delta vs FH | t50 | t100raw | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fh` | 2340 | `27.63%` | `0.00%` | `27.63%` | `0.00%` | `27.63%` | `0.04%` | `64.70%` |
| `fc` | 2340 | `27.78%` | `0.20%` | `27.78%` | `0.00%` | `27.78%` | `0.41%` | `65.00%` |
| `di` | 2340 | `22.72%` | `-6.79%` | `22.72%` | `0.00%` | `22.72%` | `-16.34%` | `65.00%` |
| `fa` | 2340 | `22.53%` | `-7.06%` | `22.53%` | `0.00%` | `22.53%` | `-16.22%` | `65.00%` |
| `fb` | 2340 | `22.59%` | `-6.97%` | `22.59%` | `0.00%` | `22.59%` | `-16.27%` | `65.00%` |
| `floor` | 2340 | `0.00%` | `-38.19%` | `0.00%` | `0.00%` | `0.00%` | `-0.00%` | `0.00%` |

### Validation past-only proxy signal

| candidate | positive gain rate | mean gain vs FH | corr endpoint-floor | corr endpoint-FH | corr path length | corr min distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fc` | `1.24%` | `0.000778` | `0.000740744901740501` | `0.478572955359064` | `-0.00939092585119528` | `-0.023219938463140705` |
| `di` | `20.90%` | `-0.026511` | `-0.3162316711368481` | `-0.4300152075137253` | `-0.0816614225055394` | `-0.029125782972543382` |
| `fa` | `21.50%` | `-0.027562` | `-0.30896703616158383` | `-0.42412955308752326` | `-0.0811656143003866` | `-0.022488379911021043` |
| `fb` | `21.03%` | `-0.027217` | `-0.31160198663461813` | `-0.42761549869817994` | `-0.08163861492866673` | `-0.024441266022727302` |

## `UCY|100`

- rows: `1440`
- diagnostic oracle improvement vs floor: `28.99%`
- diagnostic oracle improvement vs FH: `2.74%`
- low-margin share: `{'0.01': 0.8631944444444445, '0.025': 0.8916666666666667, '0.05': 0.9027777777777778}`
- oracle candidate distribution: `{'fh': 547, 'fc': 9, 'di': 620, 'fa': 106, 'fb': 0, 'floor': 158}`
- root causes: `['oracle_label_low_margin_ambiguous']`

### Validation candidate metrics

| candidate | rows | all vs floor | delta vs FH | t50 | t100raw | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fh` | 1440 | `26.99%` | `0.00%` | `0.00%` | `26.99%` | `26.99%` | `-3.45%` | `75.00%` |
| `fc` | 1440 | `27.00%` | `0.01%` | `0.00%` | `27.00%` | `27.00%` | `-3.54%` | `75.00%` |
| `di` | 1440 | `27.56%` | `0.78%` | `0.00%` | `27.56%` | `27.56%` | `-2.18%` | `75.00%` |
| `fa` | 1440 | `27.63%` | `0.87%` | `0.00%` | `27.63%` | `27.63%` | `-2.40%` | `75.00%` |
| `fb` | 1440 | `27.57%` | `0.79%` | `0.00%` | `27.57%` | `27.57%` | `-2.18%` | `75.00%` |
| `floor` | 1440 | `0.00%` | `-36.97%` | `0.00%` | `0.00%` | `0.00%` | `-0.00%` | `0.00%` |

### Test candidate metrics

| candidate | rows | all vs floor | delta vs FH | t50 | t100raw | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fh` | 1440 | `26.99%` | `0.00%` | `0.00%` | `26.99%` | `26.99%` | `-3.45%` | `75.00%` |
| `fc` | 1440 | `27.00%` | `0.01%` | `0.00%` | `27.00%` | `27.00%` | `-3.54%` | `75.00%` |
| `di` | 1440 | `27.56%` | `0.78%` | `0.00%` | `27.56%` | `27.56%` | `-2.18%` | `75.00%` |
| `fa` | 1440 | `27.63%` | `0.87%` | `0.00%` | `27.63%` | `27.63%` | `-2.40%` | `75.00%` |
| `fb` | 1440 | `27.57%` | `0.79%` | `0.00%` | `27.57%` | `27.57%` | `-2.18%` | `75.00%` |
| `floor` | 1440 | `0.00%` | `-36.97%` | `0.00%` | `0.00%` | `0.00%` | `-0.00%` | `0.00%` |

### Validation past-only proxy signal

| candidate | positive gain rate | mean gain vs FH | corr endpoint-floor | corr endpoint-FH | corr path length | corr min distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fc` | `1.11%` | `0.000080` | `0.024398431584652224` | `0.06350810755451843` | `0.013947023689959943` | `-0.005337255812486215` |
| `di` | `51.60%` | `0.007724` | `0.21379871661280583` | `-0.19781737497259563` | `0.1363462380986179` | `-0.019417604593063513` |
| `fa` | `52.78%` | `0.008545` | `0.22151781143941424` | `-0.18700202637517088` | `0.14503351363232503` | `-0.02640121138709258` |
| `fb` | `51.60%` | `0.007756` | `0.2151882894549009` | `-0.19729040654592792` | `0.13749850174813666` | `-0.019742461957297824` |

## Interpretation

- This is not a policy promotion step. It is a fresh weak-horizon diagnostic that explains why FK did not unlock uniform horizon claims.
- Diagnostic oracle rows use future labels only for upper-bound analysis; no deployed policy uses future labels.
- Uniform horizon robustness remains blocked until a validation-selected row-level horizon specialist repairs TrajNet|100, UCY|50, and UCY|100 on test without easy/proximity regressions.
- No Stage5C, SMC, true-3D, metric, or seconds-level claim is made.
