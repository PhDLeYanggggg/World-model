"""Replay the first paired risk heads from initialization, not new candidates."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_fixed_floor_tail as run


def main():
    run.torch.set_num_threads(4); run.torch.set_num_interop_threads(1)
    cfg,data,jobs,old_identity,parent_identity,bound=run.load()
    c=next(run.parent.contexts(data,jobs,old_identity))
    name,fit,held,y,pr,old,identity=run.prepare(c,data,0,parent_identity,bound)
    references=[]
    for arm in cfg['arms']:
        reference=run.PRIVATE/'heads'/(name+'_'+arm)
        record=run.done(reference/'complete.json',identity)
        home=run.PRIVATE/'training_replay'/(name+'_'+arm)
        if not (home/'checkpoint.pt').exists():
            run.api.fit(c['x'][fit],y,data['sites'][c['ids'][fit]],c['env'][fit],pr,
                seed=identity['seed'],arm=arm,settings=cfg['head_training'],identity=identity,directory=home,
                heartbeat=lambda **kw:run.beat(head='verification_'+name+'_'+arm,**kw))
        a=run.torch.load(home/'checkpoint.pt',map_location='cpu',weights_only=False)
        b=run.torch.load(reference/'checkpoint.pt',map_location='cpu',weights_only=False)
        for key in ('identity','settings','arm','seed','preprocess','tail','model','initial_model',
                    'optimizer','sampler_rng','torch_rng','draws','step','trace','mean_envelope'):
            run.exact(a[key],b[key])
        references.append(dict(arm=arm,replay=run.artifact(home/'checkpoint.pt'),
                               reference=record['artifacts']['checkpoint']))
    run.immutable_json(run.PUBLIC/'training_replay.json',dict(exact=True,heads=2,updates=4000,
        paired_initialization_optimizer_rng_draws_trace_exact=True,
        continuous_first_MSE_matches_original_pilot_resume=True,
        checkpoints=references,verification_only_not_new_candidates=True))
    print('Both full fitting replays match parameters, optimizer, RNG, draws and loss traces.')


if __name__=='__main__': main()
