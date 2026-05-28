#!/usr/bin/env bash
set -euo pipefail

# Stage42-IN t50 source-specialist reviewer replay.
# Replays evidence only. No training, no threshold search, no Stage5C, no SMC.

.venv-pytorch/bin/python run_stage42_t50_ensemble_ucy_specialist_integration.py
.venv-pytorch/bin/python run_stage42_t50_ucy_specialist_claim_audit.py
.venv-pytorch/bin/python run_stage42_t50_source_specialist_policy_freeze.py
.venv-pytorch/bin/python -m pytest tests/test_stage42_t50_ensemble_ucy_specialist_integration.py tests/test_stage42_t50_ucy_specialist_claim_audit.py tests/test_stage42_t50_source_specialist_policy_freeze.py tests/test_stage42_t50_source_specialist_reviewer_replay.py
