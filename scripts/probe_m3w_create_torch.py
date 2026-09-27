"""Allocated-node CPU optimizer/resume probe; synthetic data, no research result."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True); a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Do not train on a login node')
    import torch
    import numpy as np
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    output = Path(a.output)
    if output.exists():
        raise ValueError('Do not overwrite an existing runtime receipt')
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    torch.manual_seed(17); x = torch.randn(128, 20); y = torch.randn(128, 3)

    def initialize():
        torch.manual_seed(29)
        m = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.SiLU(), torch.nn.Linear(64, 3))
        return m, torch.optim.AdamW(m.parameters(), lr=.0003)

    def advance(model, opt, count):
        values = []
        for _ in range(count):
            opt.zero_grad(set_to_none=True); loss = (model(x)-y).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Runtime loss not finite')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.step(); values.append(float(loss.detach()))
        return values

    full, full_opt = initialize(); full_loss = advance(full, full_opt, 100)
    resumed, resumed_opt = initialize(); resumed_loss = advance(resumed, resumed_opt, 40)
    checkpoint = output.with_suffix('.pt')
    torch.save(dict(model=resumed.state_dict(), optimizer=resumed_opt.state_dict(), rng=torch.get_rng_state()), checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    resumed, resumed_opt = initialize()
    resumed.load_state_dict(state['model']); resumed_opt.load_state_dict(state['optimizer']); torch.set_rng_state(state['rng'])
    resumed_loss += advance(resumed, resumed_opt, 60)
    assert full_loss == resumed_loss
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in full.state_dict().items())
    receipt = dict(result_source='fresh_run_synthetic_cpu_optimizer_resume_probe', research_training=False,
        job_id=os.environ['SLURM_JOB_ID'], pid=os.getpid(), machine=platform.machine(), python=sys.version,
        torch=torch.__version__, numpy=np.__version__, cpu_threads=4, interop_threads=1, workers=0,
        synthetic_steps=200, checkpoint_resume_exact=True, first_loss=full_loss[0], last_loss=full_loss[-1],
        seconds=time.monotonic()-start, cuda_probed=False, mps_probed=False)
    output.write_text(json.dumps(receipt, indent=2)+'\n'); print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
