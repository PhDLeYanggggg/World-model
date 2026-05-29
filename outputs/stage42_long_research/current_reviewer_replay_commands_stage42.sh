#!/usr/bin/env bash
set -euo pipefail

# Stage42-JU current reviewer replay sequence.
# This replays evidence and claim boundaries only; it does not train models, execute Stage5C, or enable SMC.

.venv-pytorch/bin/python run_stage42_source_level_row_cache_integration.py
.venv-pytorch/bin/python run_stage42_source_level_row_cache_mechanism_audit.py
.venv-pytorch/bin/python run_stage42_source_level_incremental_ablation.py
.venv-pytorch/bin/python run_stage42_source_context_gain_harm_closure.py
.venv-pytorch/bin/python run_stage42_current_module_claim_refresh.py
.venv-pytorch/bin/python -m pytest tests/test_stage42_source_level_row_cache_integration.py tests/test_stage42_source_level_row_cache_mechanism_audit.py tests/test_stage42_source_level_incremental_ablation.py tests/test_stage42_source_context_gain_harm_closure.py tests/test_stage42_current_module_claim_refresh.py tests/test_stage42_current_reviewer_replay_package.py
