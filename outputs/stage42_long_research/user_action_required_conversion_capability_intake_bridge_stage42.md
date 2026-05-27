# User Action Required: Stage42-GE Conversion Capability

- Open `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`.
- Inspect `conversion_capability_prefill` before choosing which source to confirm.
- UCY currently has the strongest source-CV-capable after-terms plan; ETH has calibrated sources but insufficient independent sources for source-CV by itself.
- These are dry-run technical hints only. Fill `user_confirmation` manually after official terms/source identity verification.
- Then rerun:

```bash
.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py
.venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py
.venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py
.venv-pytorch/bin/python run_stage42_source_support_closure_audit.py
```
