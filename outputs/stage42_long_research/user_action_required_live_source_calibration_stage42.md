# User Action Required: Stage42-GA Source / Calibration

- No new download, conversion, or evaluation was executed.
- Local files are insufficient for new claims without source terms/path/source-identity confirmation.

| target | official URL | action |
| --- | --- | --- |
| `sdd` | https://cvgl.stanford.edu/projects/uav_data/ | Do not relabel as metric/seconds-level; next useful action is homography/scale/FPS verification only. |
| `opentraj_toolkit` | https://github.com/crowdbotp/OpenTraj | Confirm official terms and fill `opentraj_toolkit` fields in source_terms_confirmation_template_stage42.json: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity. |
| `eth_biwi_original` | https://vision.ee.ethz.ch/datsets.html | Confirm official terms and fill `eth_biwi_original` fields in source_terms_confirmation_template_stage42.json: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity. |
| `trajnetplusplus_official` | https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/ | Confirm official terms and fill `trajnetplusplus_official` fields in source_terms_confirmation_template_stage42.json: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity. |
| `ucy_crowd_original` | https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data | Confirm official terms and fill `ucy_crowd_original` fields in source_terms_confirmation_template_stage42.json: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity. |
| `tgsim` | https://data.transportation.gov/ | Keep as traffic diagnostic-only evidence; do not convert or report it as pedestrian/top-down world-model success. |
| `aerialmpt_or_other_topdown` | user_or_web_verified_official_url_required | Confirm official terms and fill `aerialmpt_or_other_topdown` fields in source_terms_confirmation_template_stage42.json: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity. |

## Suggested Commands After User Confirmation

```bash
.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py
.venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py
.venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py
.venv-pytorch/bin/python run_stage42_source_support_closure_audit.py
```
