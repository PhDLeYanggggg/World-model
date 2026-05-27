# User Action Required: Stage42-GD Calibration Hints

- Open `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`.
- Inspect `calibration_prefill` for H/FPS/stride hints before choosing which source to confirm.
- UCY and ETH rows have metric/time subset hints, but those hints are not claims until official terms/source identity and calibration validation pass.
- After user confirmation, rerun:

```bash
.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py
.venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py
.venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py
.venv-pytorch/bin/python run_stage42_source_support_closure_audit.py
```
