# Stage42 Pytest Status

- source: `fresh_run`
- command: `python3 -m pytest tests`
- result: `278 passed in 61.98s`
- targeted: `python3 -m pytest tests/test_stage42_sequence_ablation.py` -> `3 passed in 1.80s`
- targeted: `python3 -m pytest tests/test_stage42_retrained_ablation.py` -> `3 passed in 1.37s`
- targeted: `python3 -m pytest tests/test_stage42_paper_package.py` -> `3 passed in 1.36s`
- targeted: `python3 -m pytest tests/test_stage42_safety_floor.py` -> `3 passed in 1.48s`
- targeted: `python3 -m pytest tests/test_stage42_safety_floor.py tests/test_stage42_causal_ablation.py` -> `6 passed in 3.23s`
- targeted: `python3 -m pytest tests/test_stage42_causal_ablation.py` -> `3 passed in 1.35s`
- targeted: `python3 -m pytest tests/test_stage42_causal_ablation.py tests/test_stage42_full_waypoint_dynamics.py tests/test_stage42_external_validation.py` -> `11 passed in 2.16s`
- targeted: `python3 -m pytest tests/test_stage42_external_validation.py` -> `4 passed in 1.56s`
- targeted: `python3 -m pytest tests/test_stage42_full_waypoint_dynamics.py` -> `4 passed in 1.77s`
- targeted: `python3 -m pytest tests/test_stage42_full_waypoint_dynamics.py tests/test_stage42_external_validation.py` -> `8 passed in 1.38s`
- note: `.venv-pytorch/bin/python` was used for the Stage42 audit, external validation, full-waypoint dynamics, causal ablation, and safety-floor scripts. The local `.venv-pytorch` environment does not include pytest, so tests were run with the repository's existing `python3 -m pytest` test environment.
