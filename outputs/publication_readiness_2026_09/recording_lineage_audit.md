# M3W Recording and Teacher Lineage Audit

Fresh content audit. No retraining, no prediction replay, no historical output replacement.

- Recording boundary pass: False
- Inherited teacher boundary pass: False
- Training with unchanged caches allowed: False

## Byte-Identical Source Files

- SHA256 `2697ac758db7b92f31871552f39820d0f5f3328994f33a170fcd4cb6df93f403`; splits: ['train']
  - `external_data/OpenTraj/datasets/TrajNet/Test/crowds/students002.txt`: {'train': 765}
  - `external_data/OpenTraj/datasets/TrajNet/Test/crowds/uni_examples.txt`: {'train': 765}
- SHA256 `6ce35fe5215897674a5e12f2e56442fac77f9ec03ff58c324876e7a9c77882c2`; splits: ['train', 'val']
  - `external_data/OpenTraj/datasets/TrajNet/Train/crowds/students001.txt`: {'train': 47223}
  - `external_data/OpenTraj/datasets/UCY/students01/students001-trajnet.txt`: {'val': 47223}
- SHA256 `c31269a2b21b1152cf22be3aa6d0ae5b0b71117f22981f0ae0c1b30b64146797`; splits: ['test', 'train']
  - `external_data/OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt`: {'test': 9540}
  - `external_data/OpenTraj/datasets/UCY/zara03/crowds_zara03.txt`: {'train': 9540}

## Numeric Recording Overlap

Frame/agent/x/y identities rounded to 5 decimal places. No unit conversion or trajectory realignment.

| source pair | shared raw rows | fraction of smaller | identical cached windows | crosses splits |
| --- | ---: | ---: | ---: | --- |
| `external_data/OpenTraj/datasets/TrajNet/Test/crowds/students002.txt` / `external_data/OpenTraj/datasets/TrajNet/Test/crowds/uni_examples.txt` | 680 | 1.0000 | 765 | False |
| `external_data/OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt` / `external_data/OpenTraj/datasets/UCY/zara03/crowds_zara03.txt` | 3600 | 1.0000 | 9540 | True |
| `external_data/OpenTraj/datasets/TrajNet/Train/crowds/students001.txt` / `external_data/OpenTraj/datasets/UCY/students01/students001-trajnet.txt` | 17820 | 1.0000 | 47223 | True |

## Teacher Exposure

Old split counts within each new split: `{'train': {'test': 9540, 'train': 63602, 'val': 73667}, 'val': {'test': 47223, 'train': 17070, 'val': 37153}, 'test': {'test': 9540, 'train': 78270, 'val': 1926}}`.

New val/test rows that were legacy teacher-training rows: `{'val': 17070, 'test': 78270}`.

Inherited old_split metadata available: `{'train': True, 'val': True, 'test': True}`. For raw Stage35 caches, the supplied split is used; no inherited teacher audit is claimed.

Stage35 fits supervised models on its old train split. Reusing those outputs in newly assigned val/test does not provide a held-out teacher. Some old-train rows also reach new train, so its cached teacher predictions are not out-of-fold gain/harm supervision.

Stage37 also ranks selector variants using test metrics (src/stage37_t50_history.py:t50_selector). Its reported bootstrap interval does not correct test-based selection bias.

## Horizon Alignment

| split | requested horizon | rows | exact frame delta | min actual | max actual |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 10 | 46716 | 39238 | 10.0 | 12.0 |
| train | 25 | 41248 | 0 | 30.0 | 30.0 |
| train | 50 | 35467 | 30393 | 50.0 | 54.0 |
| train | 100 | 23378 | 20764 | 100.0 | 102.0 |
| val | 10 | 32706 | 32706 | 10.0 | 10.0 |
| val | 25 | 28703 | 0 | 30.0 | 30.0 |
| val | 50 | 24741 | 24741 | 50.0 | 50.0 |
| val | 100 | 15296 | 15296 | 100.0 | 100.0 |
| test | 10 | 26132 | 26131 | 10.0 | 20.0 |
| test | 25 | 23780 | 0 | 26.0 | 40.0 |
| test | 50 | 21754 | 21753 | 50.0 | 60.0 |
| test | 100 | 18070 | 18069 | 100.0 | 110.0 |

## Consequence and Repair

Distinct dataset labels or file paths do not establish independent recordings. The affected Stage43/44 generalization results are exploratory, not clean cross-dataset confirmation. This audit does not reproduce or invalidate every SDD experiment.

Group aliases by original recording, retain one canonical source, quarantine unresolved aliases, and declare scene/recording-held-out protocols before training. Rebuild goals, baselines, normalizers and supervised teachers exclusively within each new training fold. Do not reuse legacy target-derived labels/features as a supposedly held-out floor.

All local historic evaluation sources are already exposed to research. A repaired split is not an untouched confirmation set; prospectively frozen replication or a new source is still required.

Dataset-local/raw-frame only. No true-3D, foundation, metric or seconds claim. Stage5C and SMC remain disabled.
