# User Action Required: Stage42-EM Official Source Links

No dataset is conversion-ready yet. The agent cannot accept terms, infer allowed use, or treat toolkit/local files as data permission.

| priority | dataset | official candidates | required user action |
| ---: | --- | --- | --- |
| 1 | `ucy_crowd_original` | https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data<br>https://graphics.cs.ucy.ac.cy/portfolio | Open the UCY crowd-data page, confirm official terms/allowed use, then fill local_path and source_identity for UCY_students03 / UCY_zara01 / UCY_zara02. |
| 2 | `eth_biwi_original` | https://vision.ee.ethz.ch/datsets.html | Open the ETH CVL dataset page, confirm research-only terms and local BIWI ETH/Hotel source identity before guarded conversion. |
| 3 | `aerialmpt_or_other_topdown` | user_or_web_verified_official_url_required | Provide an official HTTPS source page, terms/license, local path, and source identity before any conversion. |
| 4 | `opentraj_toolkit` | https://github.com/crowdbotp/OpenTraj | Use OpenTraj for loaders/toolkit context only after confirming each underlying dataset's official terms and source identity. |
| 5 | `trajnetplusplus_official` | https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/<br>https://www.epfl.ch/labs/vita/datasets/<br>https://github.com/vita-epfl/trajnetplusplusbaselines | Confirm the official TrajNet++ data/challenge access path and underlying dataset terms, then provide a local official/source-identified copy. |

After filling the intake template, run:

```bash
.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py
.venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py
```
