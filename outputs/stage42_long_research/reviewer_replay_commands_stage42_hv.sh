#!/usr/bin/env bash
set -euo pipefail

# Stage42-HW replay evidence tier refresh commands.
# These commands replay/audit evidence only; they do not train, tune thresholds, execute Stage5C, or enable SMC.

.venv-pytorch/bin/python run_stage42_group_consistency_t100_easy_guard_runtime.py
.venv-pytorch/bin/python run_stage42_t100_runtime_batch_replay_sufficiency.py
.venv-pytorch/bin/python run_stage42_t100_runtime_row_cache_replay.py
.venv-pytorch/bin/python -m pytest tests/test_stage42_group_consistency_t100_easy_guard_runtime.py tests/test_stage42_t100_runtime_batch_replay_sufficiency.py tests/test_stage42_t100_runtime_row_cache_replay.py
