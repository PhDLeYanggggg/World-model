# User Action Required: Stage42-IA HZ to CG Intake Bridge

Bridged candidate intake: `outputs/stage42_long_research/source_terms_confirmation_intake_from_hz_stage42.json`

This bridge did not activate conversion. After confirming official source terms in the HZ template, rerun:

`.venv-pytorch/bin/python run_stage42_hz_to_cg_intake_bridge.py`

Then either explicitly select the bridged intake in a future guarded runner or copy reviewed confirmations into the canonical CG intake template. Do not convert/evaluate until validator, guarded conversion, no-leakage, and source-CV all pass.
