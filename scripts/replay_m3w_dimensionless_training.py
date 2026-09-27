"""Fresh fixed-first training replay; never a new candidate or selected seed."""
import fcntl
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_dimensionless_refit as run
import numpy as np
import torch
from src.world_model.m3w_native_forecast import fit_trial


def exact(a,b):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:
            exact(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            exact(x,y)
    else:
        assert a==b


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg,reg,data,jobs,_=run.load()
    run.endpoints(jobs)
    j=jobs[0]
    reference=run.PRIVATE/'dimensionless'/j['key']/'complete.json'
    receipt=json.loads(reference.read_text())
    control=torch.load(ROOT/receipt['checkpoint']['path'],map_location='cpu',weights_only=False)
    home=run.PRIVATE/'training_replay'/j['key']
    home.mkdir(parents=True,exist_ok=True)
    with (home/'replay.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (home/'checkpoint.pt').exists():
            raise FileExistsError('Fresh replay exists; verify its receipt instead of silently rerunning')
        model=run.make_model(j)
        report=fit_trial(model,data,j['design'],seed=j['old_identity']['seed'],settings=reg['training'],
            identity=receipt['identity'],directory=home,
            heartbeat=lambda **kw:print(json.dumps(kw),flush=True))
        checkpoint=home/'checkpoint.pt'
        new=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for k in ('identity','settings','seed','step','losses','draws','train_ids','factors','model',
                  'optimizer','sampler_rng','torch_rng'):
            exact(new[k],control[k])
        run.immutable_json(run.PUBLIC/'training_replay.json',dict(trial=j['key'],result_source='fresh_training_replay',
            steps=report['step'],fit_seconds=report['seconds'],all_parameters_exact=True,
            optimizer_exact=True,all_logged_losses_exact=True,sampler_counts_and_rng_exact=True,
            torch_rng_exact=True,reference=run.artifact(reference),checkpoint=run.artifact(checkpoint),
            fresh_geometry_conversion=False,independent_replication=False,new_candidate=False))
        print(json.dumps(dict(training_replay_exact=True,trial=j['key'],steps=report['step'])),flush=True)


if __name__=='__main__':
    main()
