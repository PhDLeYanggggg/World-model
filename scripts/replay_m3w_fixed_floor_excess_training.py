"""Full first-head fitting replay; no extra model-selection candidate."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_fixed_floor_excess as run


def main():
    run.torch.set_num_threads(4); run.torch.set_num_interop_threads(1)
    cfg,data,jobs,oid,pid,pbound,bound=run.load()
    c=next(run.floor_api.contexts(data,jobs,oid))
    name,fit,held,y,pr,old,ms,mse,identity=run.control(c,data,0,pid,pbound,bound)
    ref=run.PRIVATE/'heads'/name; rec=run.done(ref/'complete.json',identity)
    home=run.PRIVATE/'training_replay'/name
    if not (home/'checkpoint.pt').exists():
        run.api.fit(c['x'][fit],y,data['sites'][c['ids'][fit]],c['env'][fit],pr,
            seed=ms['seed'],settings=cfg['head_training'],identity=identity,directory=home,
            heartbeat=lambda **kw:run.beat(head='verification_'+name,**kw))
    a=run.torch.load(home/'checkpoint.pt',map_location='cpu',weights_only=False)
    b=run.torch.load(ref/'checkpoint.pt',map_location='cpu',weights_only=False)
    for k in ('identity','settings','objective','seed','preprocess','model','initial_model','optimizer',
              'sampler_rng','torch_rng','draws','step','trace','mean_envelope'): run.api.exact(a[k],b[k])
    run.api.assert_matched(a,ms)
    run.immutable_json(run.PUBLIC/'training_replay.json',dict(exact=True,new_heads=1,updates=2000,
        optimizer_rng_draws_loss_trace_exact=True,continuous_fit_matches_original_pilot_resume=True,
        matched_cached_MSE_control=True,replay=run.artifact(home/'checkpoint.pt'),
        reference=rec['artifacts']['checkpoint'],verification_only_not_new_candidate=True))
    print('Full signed-excess training replay and cached-control matching passed.')


if __name__=='__main__': main()
