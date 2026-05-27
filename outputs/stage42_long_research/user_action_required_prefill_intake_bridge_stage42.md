# User Action Required: Stage42-GC Prefilled Intake

- Open `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`.
- For any dataset you want to unblock, inspect the `prefill_suggestion` block.
- Only after checking official terms, copy/edit the suggested path and source identity into `user_confirmation.local_path` and `user_confirmation.source_identity`.
- Fill `terms_accepted_by_user`, `terms_acceptance_date`, `allowed_use`, `redistribution_allowed`, `derived_data_allowed`, and `confirmed_by_user` yourself.
- Then rerun:

```bash
.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py
.venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py
.venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py
.venv-pytorch/bin/python run_stage42_source_support_closure_audit.py
```
