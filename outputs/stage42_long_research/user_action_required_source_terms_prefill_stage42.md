# User Action Required: Stage42-GB Source Terms Prefill

Use this as a checklist. Do not treat the prefill draft as authorization.

- prefill draft: `outputs/stage42_long_research/source_terms_confirmation_prefill_stage42.json`
- copy a suggested local path only after verifying it is the official allowed source copy
- fill `terms_accepted_by_user`, `terms_acceptance_date`, `allowed_use`, `local_path`, and `source_identity` in `source_terms_confirmation_template_stage42.json`
- then rerun the source terms validator and guarded conversion queue

| dataset | suggested path | official URL |
| --- | --- | --- |
| `ucy_crowd_original` | `external_data/OpenTraj/datasets/UCY` | https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data |
| `eth_biwi_original` | `external_data/OpenTraj/datasets/ETH` | https://vision.ee.ethz.ch/datsets.html |
| `trajnetplusplus_official` | `external_data/OpenTraj/datasets/TrajNet` | https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/ |
| `opentraj_toolkit` | `external_data/OpenTraj` | https://github.com/crowdbotp/OpenTraj |
| `aerialmpt_or_other_topdown` | `data/aerialmpt/DLR_AerialMPT_Dataset.zip` | user_or_web_verified_official_url_required |

```bash
.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py
.venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py
.venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py
.venv-pytorch/bin/python run_stage42_source_support_closure_audit.py
```
