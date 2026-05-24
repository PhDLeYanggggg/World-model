# M3W-Neural v1 Reproducibility

Use arm64 PyTorch for training/evaluation commands on Apple Silicon.

```bash
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_build_seq2seq_dataset.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_build_all_agent_dataset.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_fresh_self_gated_endpoint_candidate.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_endpoint_geometry_audit.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_gates.py
python -m pytest tests
```

- frozen git commit at package time: `0aca3fe`
- package input hash: `017c27e874ecb9f08b5d8bf71edde0f891318e8eede796e1ad80bb3741946a4c`
- Do not commit caches/checkpoints/raw data when reproducing.
