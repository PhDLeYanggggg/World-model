# User Action Required: Stage42-GM Guarded Conversion Harness

The guarded converter did not run because `contract_ready_now = 0`.

Required sequence before any conversion:

1. Fill `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json` after official terms/path/source identity review.
2. Run `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`.
3. Run `.venv-pytorch/bin/python run_stage42_source_conversion_contract.py`.
4. Run `.venv-pytorch/bin/python run_stage42_guarded_conversion_harness.py`.
5. Only a future source-specific converter may build a feature store, and it must redo no-leakage/source-CV/metric-time guards.

Do not count post-confirmation candidates as permission, converted data, or evaluated results.
