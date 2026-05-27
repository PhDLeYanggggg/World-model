#!/usr/bin/env bash
set -euo pipefail

# Stage42-DM reviewer replay sequence.
# This script regenerates/replays evidence only; it does not train models, execute Stage5C, or enable SMC.

.venv-pytorch/bin/python run_stage42_replay_proximity_guard_policy.py
.venv-pytorch/bin/python run_stage42_batch_replay_proximity_guard_policy.py
.venv-pytorch/bin/python run_stage42_replay_group_consistency_policy.py
.venv-pytorch/bin/python run_stage42_group_consistency_runtime_policy.py
.venv-pytorch/bin/python run_stage42_module_contribution_ledger.py
.venv-pytorch/bin/python run_stage42_claim_boundary_linter.py
.venv-pytorch/bin/python run_stage42_source_action_consolidator.py
.venv-pytorch/bin/python run_stage42_evidence_provenance_verifier.py
.venv-pytorch/bin/python run_stage42_paper_freeze_candidate_manifest.py
.venv-pytorch/bin/python -m pytest tests/test_stage42_proximity_guard_policy_replay.py tests/test_stage42_proximity_guard_batch_replay.py tests/test_stage42_group_consistency_policy_replay.py tests/test_stage42_group_consistency_runtime_policy.py tests/test_stage42_module_contribution_ledger.py tests/test_stage42_claim_boundary_linter.py tests/test_stage42_source_action_consolidator.py tests/test_stage42_evidence_provenance_verifier.py tests/test_stage42_paper_freeze_candidate_manifest.py
