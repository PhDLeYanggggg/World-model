# Private checkpoint directory preflight

The first eval process PID98349 exited1 while trying to save the first
allocation checkpoint. The private `progress` parent directory did not exist;
the generic save_checkpoint helper does not create parents. No new policy
error score or aggregate group was produced. All36 trained heads remain frozen.
The original traceback is retained privately as `readout.log`.

Create the owned runtime directory before evaluation:

```sh
mkdir -p data/stage_cvpr2027_experiments/european_coupled_benefit_risk_v1/progress
.venv-pytorch/bin/python scripts/run_m3w_coupled_benefit_risk.py eval --resume
```

The resumed process uses `readout_after_progress_setup.log`. No scientific
code,config,registered hash,model,loss,split,budget or label changes. It may
recompute the first causal allocation,which had no saved checkpoint. This
repairs execution setup,not a failed research outcome. The original frozen
entry point still requires this directory preflight;do not claim it created
the missing directory itself.
