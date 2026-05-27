# Stage42-EM Official Source Link Audit

- source: `fresh_stage42_official_source_link_audit`
- generated_at_utc: `2026-05-27T02:50:20.490298+00:00`
- retrieval_date: `2026-05-27`
- git_commit: `13097e8`
- input_hash: `b539b13687576e1595ae7aee7b5f1d5cd98eb7dcf97ba9a96252244cf6af8955`
- gate: `14 / 14`
- verdict: `stage42_em_official_source_link_audit_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-EM only resolves official source-link and manual confirmation gaps; it does not download, convert, train, or evaluate data.
- Official source links are not legal acceptance; user must still confirm terms, allowed use, local path, and source identity.
- local path, parseability, toolkit mirrors, or GitHub code licenses are not blanket permission for third-party datasets.
- future endpoints / waypoints are labels/eval only, never inference inputs.
- No central velocity, no test endpoints for goals, no test-threshold tuning.
- t+50 / t+100 remain raw-frame horizons; no seconds-level claim.
- dataset-local/raw-frame coordinates are not global metric coordinates.
- Stage5C latent generative was not executed.
- SMC was not enabled.

## Summary

- targets audited: `5`
- official/toolkit source candidates recorded: `4`
- manual terms required targets: `5`
- auto download allowed now: `0`
- conversion ready now: `0`
- converted/evaluated now: `0` / `0`
- estimated t50/t100 after terms: `10060` / `5696`

## Official Source Rows

| dataset | confidence | official candidates | manual terms | conversion ready now | action |
| --- | --- | --- | ---: | ---: | --- |
| `ucy_crowd_original` | `official_primary_plus_official_lab_portfolio` | https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data<br>https://graphics.cs.ucy.ac.cy/portfolio | True | False | Open the UCY crowd-data page, confirm official terms/allowed use, then fill local_path and source_identity for UCY_students03 / UCY_zara01 / UCY_zara02. |
| `eth_biwi_original` | `official_primary` | https://vision.ee.ethz.ch/datsets.html | True | False | Open the ETH CVL dataset page, confirm research-only terms and local BIWI ETH/Hotel source identity before guarded conversion. |
| `aerialmpt_or_other_topdown` | `missing_official_source` | user_or_web_verified_official_url_required | True | False | Provide an official HTTPS source page, terms/license, local path, and source identity before any conversion. |
| `opentraj_toolkit` | `official_github_toolkit_only` | https://github.com/crowdbotp/OpenTraj | True | False | Use OpenTraj for loaders/toolkit context only after confirming each underlying dataset's official terms and source identity. |
| `trajnetplusplus_official` | `official_project_plus_official_github_code` | https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/<br>https://www.epfl.ch/labs/vita/datasets/<br>https://github.com/vita-epfl/trajnetplusplusbaselines | True | False | Confirm the official TrajNet++ data/challenge access path and underlying dataset terms, then provide a local official/source-identified copy. |

## User Action

Fill the Stage42-EH intake template only after manually checking the official source terms and local source identity:

- `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`
- validator: `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`
- guarded launcher after validator readiness: `.venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py`

## Gate

| gate | pass |
| --- | ---: |
| `intake_template_exists` | True |
| `readiness_manifest_checked` | True |
| `targets_audited` | True |
| `official_candidates_recorded` | True |
| `ucy_has_official_url` | True |
| `eth_has_official_url` | True |
| `trajnet_has_official_url` | True |
| `manual_terms_preserved` | True |
| `no_auto_download` | True |
| `no_conversion_or_eval` | True |
| `user_action_written` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
