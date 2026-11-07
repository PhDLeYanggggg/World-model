# Cross-Archive Source Check Completed

I screened the complete available MovingDroneCrowd A/B image roster against
the DroneCrowd archive before assigning external evaluation roles. The screen
covers 34 A recordings and 24 exposed B recordings, with 10 shared recording
IDs, for 48 unique old recordings. All 34 actual A actor recording IDs were
checked through the original numeric-ID to recording-name bridge.

| Frozen check | Result |
|---|---:|
| Existing development recordings | 48 |
| Prospective DroneCrowd recordings | 112 |
| Cross-archive recording pairs | 5,376 |
| Three-view frame comparisons | 48,384 |
| Strong matches | 0 |
| Ambiguous matches | 0 |
| Additional quarantined exclusion groups | 0 |

No pair meets the frozen matching criteria in the three cached views per
recording. This is stronger evidence than comparing dataset names alone, but
does not show that every recording is a distinct physical site. The prior
within-DroneCrowd result is unchanged: 45 conservative exclusion groups,
including cross-folder relations, so the original release split is not used.
The result is scoped to the specified A/B roster and available image views.

## Two Prospective Roles, Neither Assigned

| Option | Calibration groups / clips | Confirmation groups / clips |
|---|---:|---:|
| Dedicated calibration, then one confirmation | 15 / 40 | 30 / 72 |
| Fully frozen zero-shot external evaluation | 0 / 0 | 45 / 112 |

Both alternatives preserve whole exclusion groups and retain the original 2%
conditions. Group ordering is fixed by an ID hash, not selected using target
availability, prediction errors, gains or support. The first option can test
calibration without reusing confirmation scenes. The second tests frozen
transport without adapting on this archive. Both keep model fitting and model
selection on existing development data. Neither alternative is a signed role
manifest or a permission to read external prediction outcomes.
[Machine-readable alternatives](../dronecrowd_external_role_proposal_v1/alternatives.json).

## Reproduction And Actual Execution

- Fresh matching: PID 7093 completes the 112-pair timing pilot in 23.517 s.
  PID 7168 resumes from that receipt and completes the remaining 47 old
  recordings in 1,079.322 s, peak RSS 1,810,989,056 B. Both exit 0.
- Cached verification: PID 8452 rechecks the input hashes and all 48 saved
  matching receipts, reproducing the complete analysis exactly in 0.393 s
  after setup. This is not a second 48,384-pair match run. No new image download.
- Fresh role-proposal calculation: both alternatives regenerate exactly. The
  input bridge separately checks all 112 recordings and 336 past prefixes;
  no label method or forecast inference runs. Thirteen scoped tests pass.
- Local computation was suitable for screening. A fresh read-only CREATE
  queue check timed out after 60 s; remote state remains unknown. No remote
  job submitted or changed. Full external inference would use streaming rather
  than storing a dense feature table.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_dronecrowd_cross_archive.py screen --max-new-recordings 1
.venv-pytorch/bin/python scripts/audit_m3w_dronecrowd_cross_archive.py screen
.venv-pytorch/bin/python scripts/audit_m3w_dronecrowd_cross_archive.py verify
.venv-pytorch/bin/python scripts/propose_m3w_dronecrowd_roles.py
.venv-pytorch/bin/python scripts/check_m3w_dronecrowd_frozen_inputs.py
```

This is preparation for an external experiment, not evidence that the method
generalizes. The original recording-risk results stay failed where reported.
The offline 8/12 native annotation-stride task stays image-space. No new
metric/time, true-3D, foundation, deployment or submission claim. Stage5C and
SMC are off. Raw images, descriptors and recording caches are not committed.
