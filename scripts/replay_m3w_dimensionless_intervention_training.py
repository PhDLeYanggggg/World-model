"""Fixed-first full head replay, including the original pilot/resume boundary."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_dimensionless_intervention as run
from scripts.replay_m3w_dimensionless_training import exact
import numpy as np


def main():
    run.torch.set_num_threads(4); run.torch.set_num_interop_threads(1)
    cfg, data, jobs, identity, _ = run.load()
    j=jobs[0]; controller=1; candidate='dimensionless'; task='utility'
    ids, cvp, bank=run.candidates(j,data)
    x, env, _=run.inference_features(data['geometry'][ids],cvp,bank[candidate])
    roles=run.role_indices(data['sites'],identity['rosters'],0,controller)
    fit_ids=roles['controller']; pos=np.searchsorted(ids,fit_ids)
    cv=data['baseline_ade'][fit_ids,1]
    err=run.native_errors(bank[candidate][pos].astype(float)+data['origin'][fit_ids,None],
        data['target_eval'][fit_ids],data['valid'][fit_ids],np.ones(len(fit_ids)))[0]
    y,_=run.relative_targets(cv,cv,err,reference='cv',event='all',easy_cut=j['design']['easy_cut'])
    pr=run.preprocess(x[pos],y,cv,data['sites'][fit_ids],'__excluded_readout__')
    name=run.group_name(j,controller,candidate)+'_'+task
    rec=run.read_done(run.PRIVATE/'heads'/name/'complete.json',identity)
    home=run.PRIVATE/'training_replay'/name
    if (home/'checkpoint.pt').exists():raise FileExistsError('Replay already exists; verify instead of retraining')
    kwargs=dict(seed=17,task=task,settings=cfg['head_training'],identity=rec['identity'],directory=home,
                heartbeat=lambda **kw:run.beat(head='verification_'+name,**kw))
    run.head.fit(x[pos],y,data['sites'][fit_ids],env[pos],pr,stop_at=100,**kwargs)
    _, fit=run.head.fit(x[pos],y,data['sites'][fit_ids],env[pos],pr,resume=True,**kwargs)
    a=run.torch.load(home/'checkpoint.pt',map_location='cpu',weights_only=False)
    b=run.torch.load(ROOT/rec['artifacts']['checkpoint']['path'],map_location='cpu',weights_only=False)
    for k in ('identity','settings','task','seed','preprocess','mean_envelope','model','optimizer',
              'sampler_rng','torch_rng','draws','step','trace'):exact(a[k],b[k])
    run.immutable_json(run.PUBLIC/'training_replay.json',dict(result_source='fresh_verification_only_training',
        head=name,steps=fit['step'],fit_seconds=fit['seconds'],parameters_optimizer_rng_draws_losses_exact=True,
        replay_pid=run.os.getpid(),checkpoint=run.artifact(home/'checkpoint.pt'),
        reference=rec['artifacts']['checkpoint'],new_independent_model=False,cold_raw_rebuild=False))


if __name__=='__main__':main()
