# M3W-Neural v1 Reproducibility

Use arm64 PyTorch for training/evaluation commands on Apple Silicon.

```bash
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_build_seq2seq_dataset.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_build_all_agent_dataset.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_bounded_neural_blend_dynamics.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_composite_tail_evidence.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_composite_tail_multiseed.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_pure_ucy_source_validation.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_pure_ucy_neural_retrain.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_pure_ucy_neural_statistical_evidence.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_endpoint_to_full_trajectory_repair.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_endpoint_to_full_statistical_evidence.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_calibrated_shape_meta_policy.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_endpoint_geometry_audit.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_gates.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_freeze_m3w_neural_v1.py
python -m pytest tests
```

- frozen git commit at package time: `704bdc3`
- package input hash: `fc9c10f0a98a255008eefc4b0e8cccabf5da069d9bd9623cb36b630976c462cf`
- Do not commit caches/checkpoints/raw data when reproducing.
