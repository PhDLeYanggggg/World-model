# User Action Required: Stage42-GF Post-Confirmation Conversion Plan

Open `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json` and inspect each row's `conversion_capability_prefill` before filling `user_confirmation`.

Highest-value sources after terms confirmation:

| rank | dataset | source | t50 | t100 | why |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | `ucy_crowd_original` | `UCY_students03` | 6491 | 3413 | source-CV after terms |
| 2 | `ucy_crowd_original` | `UCY_zara02` | 2823 | 2095 | source-CV after terms |
| 3 | `ucy_crowd_original` | `UCY_zara01` | 240 | 97 | source-CV after terms |
| 4 | `ucy_crowd_original` | `UCY_zara03` | 0 | 0 | source-CV after terms |
| 5 | `eth_biwi_original` | `ETH_seq_eth` | 291 | 91 | technical source-specific candidate |
| 6 | `eth_biwi_original` | `ETH_seq_hotel` | 215 | 0 | technical source-specific candidate |

Required user-confirmed fields before any conversion:

- `terms_accepted_by_user`
- `terms_acceptance_date`
- `allowed_use`
- `redistribution_allowed`
- `derived_data_allowed`
- `local_path`
- `source_identity`
- `confirmed_by_user`

Then rerun:

```bash
.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py
.venv-pytorch/bin/python run_stage42_post_confirmation_conversion_plan.py
.venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py
```
