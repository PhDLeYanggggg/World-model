# Stage42 Pytest Status

- source: `fresh_run`
- command: `python3 -m pytest tests`
- result: `263 passed in 69.16s`
- targeted: `python3 -m pytest tests/test_stage42_external_validation.py` -> `4 passed in 1.56s`
- targeted: `python3 -m pytest tests/test_stage42_full_waypoint_dynamics.py` -> `4 passed in 1.77s`
- targeted: `python3 -m pytest tests/test_stage42_full_waypoint_dynamics.py tests/test_stage42_external_validation.py` -> `8 passed in 1.38s`
- note: `.venv-pytorch/bin/python` was used for the Stage42 audit, external validation, and full-waypoint dynamics scripts. The local `.venv-pytorch` environment does not include pytest, so tests were run with the repository's existing `python3 -m pytest` test environment.
