# Approved DroneCrowd Calibration And Confirmation Roles

The author selected the existing prospective 15/30 alternative on
7 November 2026. The [v2 role manifest](../../../configs/m3w_dronecrowd_calibration_confirmation_v2.json)
supersedes only the DroneCrowd roles in the older all-confirmation
reservation. The older file remains available to verify earlier studies;
it is no longer the role authority for this experiment. Other datasets
keep their existing assignments.

| Role | Exclusion groups | Recordings | Input queries |
| --- | ---: | ---: | ---: |
| Threshold calibration | 15 | 40 | 11,240 |
| One independent confirmation | 30 | 72 | 20,232 |

The recorded split uses the same label-blind group list, without moving
any recording after inspecting an outcome. Group and recording sets
are disjoint and exhaust the 112 reserved recordings. The input counts
come from the previously replayed preflight; they are not independent
samples or known easy-reference counts. Independent here means separate
exclusion groups, not verified physical sites.

The comparison remains native observation-8 / prediction-12 at stride
one in image coordinates. The original 2%, reference-support and
undefined-risk rules are unchanged. The role overlay authorizes no
outcome readout by itself. Next freeze the exact calibration candidates,
threshold procedure, and fitting-derived cuts; use only the 15-group
calibration role to choose a threshold. The 30-group confirmation
outcomes stay unopened until the selected policy and one-shot readout
are frozen. Do not recast calibration outcomes as confirmation or move
recordings between the groups.

```sh
.venv-pytorch/bin/python scripts/freeze_m3w_dronecrowd_calibration_roles.py verify
```

Verification reads JSON receipts only, not annotations or predictions.
No external scores, metric/time claim, deployment change, Stage5C or SMC.
