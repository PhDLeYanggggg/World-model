# Stage43 Locked Candidate Reproducibility Checklist

- input hash: `6de9b0b9df474351a3af57ed48f8eefd5436c7afcac958f3c69ccfeafa432d63`
- git commit at generation: `b0430d0`
- runtime: `.venv-pytorch/bin/python` on arm64 Apple Silicon path where training is needed.
- DataLoader multiprocessing must remain off for local training paths.
- This refresh itself does not run new training or conversion.

## Required Verification Commands

```bash
.venv-pytorch/bin/python run_stage43_locked_candidate_paper_package_refresh.py
.venv-pytorch/bin/python -m pytest tests/test_stage43_locked_candidate_paper_package_refresh.py tests/test_stage43_external_validation_matrix.py -q
.venv-pytorch/bin/python -m pytest tests
```
