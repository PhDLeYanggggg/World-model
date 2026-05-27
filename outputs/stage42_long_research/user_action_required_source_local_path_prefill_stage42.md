# User Action Required: Stage42-HY Source Local Path Confirmation

请只在你确认官方条款、允许用途、source identity、local path 都正确后，再允许后续 guarded conversion。

| dataset | domain | local path found | best path | t50 after terms | t100 after terms | remaining confirmation |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `ucy_crowd_original` | `UCY` | `True` | `external_data/OpenTraj/datasets/UCY` | 9554 | 5605 | terms_accepted_by_user, terms_acceptance_date, allowed_use |
| `eth_biwi_original` | `ETH_UCY` | `True` | `external_data/OpenTraj/datasets/ETH` | 506 | 91 | terms_accepted_by_user, terms_acceptance_date, allowed_use |
| `aerialmpt_or_other_topdown` | `other_topdown` | `True` | `data/aerialmpt` | 0 | 0 | terms_accepted_by_user, terms_acceptance_date, allowed_use |
| `opentraj_toolkit` | `OpenTraj` | `True` | `external_data/OpenTraj` | 0 | 0 | terms_accepted_by_user, terms_acceptance_date, allowed_use |
| `trajnetplusplus_official` | `TrajNet` | `True` | `external_data/OpenTraj/datasets/TrajNet++` | 0 | 0 | terms_accepted_by_user, terms_acceptance_date, allowed_use |

Required fields per source:

- `terms_accepted_by_user`: true/false
- `terms_acceptance_date`: YYYY-MM-DD
- `allowed_use`: e.g. research_only / commercial_allowed / unknown
- `local_path`: exact local path to use
- `source_identity`: official source or mirror identity; OpenTraj mirror is not automatically official-source permission

No conversion/evaluation/metric-time claim is allowed until these fields are confirmed and a guarded conversion passes.
