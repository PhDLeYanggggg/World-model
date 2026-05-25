# Stage42 Pytest Status

- source: `fresh_run`
- command: `python3 -m pytest tests`
- result: `255 passed in 65.35s`
- note: `.venv-pytorch/bin/python` was used for the Stage42 audit script. The local `.venv-pytorch` environment does not include pytest, so tests were run with the repository's existing `python3 -m pytest` test environment.
