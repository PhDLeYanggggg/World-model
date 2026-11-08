"""Frozen shape contrast under the simpler all-recipient half-radius rule."""
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_opendd_flight_shape_risk as flight
from scripts import run_m3w_opendd_causal_amplitude as run


def main():
    if platform.system()!='Darwin' or platform.machine()!='arm64':
        raise RuntimeError('Use native arm64 .venv-pytorch on this host')
    import torch
    from src.world_model import m3w_opendd_velocity_candidate as model
    from src.world_model import m3w_radius_shape as shape

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg, _ = flight.register()
    freeze = json.loads((flight.PUBLIC/'checkpoint_freeze.json').read_text())
    prev = json.loads((flight.shape_run.PUBLIC/'checkpoint_freeze.json').read_text())
    sfile = run.PUBLIC/'amplitude_spread.json'
    spread = json.loads(sfile.read_text())
    if (freeze['registration']!=reg or not spread['query_mass_exact'] or spread['decisions_changed']):
        raise ValueError('Require verified frozen training and fixed half reference')
    outer = run.shape_run.frozen.load_networks(torch,model)
    src = {s:run.shape_run.site_data(s,torch,model,shape) for s in run.SITES}
    rows = {}
    for i,s in enumerate(run.SITES):
        d = src[s]
        rows[s] = []
        cv = d['cv']
        cuts = run.base.cuts_for([src[k] for k in run.SITES if k!=s])
        for seed in run.SEEDS:
            c = run.shape_run.candidate(d,outer[seed][i],torch,model)
            nets = dict(previous_point=flight.previous_model(s,seed,torch,model,prev))
            for name in ('flight_relative','flight_noncancel'):
                path = flight.PRIVATE/f'{s}_{seed}_{name}.pt'
                if run.prior.sha(path)!=freeze['fits'][f'{s}/{seed}/{name}']['sha256']:
                    raise ValueError('Shape checkpoint changed')
                net = model.network(138)
                net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
                nets[name] = net.eval()
            full = flight.paths_for(d,c,nets,torch,shape)
            paths = {k:cv+.5*(p-cv) for k,p in full.items()}
            arms = run.base.readout(d,paths,c['r']>0,cuts)
            old = next(v['policies']['fixed_half'] for v in spread['by_location'][s] if v['seed']==seed)
            if (arms['incumbent']['passed']!=old['passed'] or
                arms['incumbent']['comparisons']['cv']!=old['vs_cv']):
                raise ValueError('Frozen half-scale reference did not replay')
            rows[s].append(dict(seed=seed,arms=arms))
    run.prior.write_json(run.PUBLIC/'half_radius_shapes.json',dict(flight_registration=reg,
        spread_sha256=run.prior.sha(sfile),posthoc_diagnostic=True,exact_recipient_radius_match=True,
        summary=run.base.summary(rows),by_location=rows,decisions_changed=False),immutable=True)


if __name__=='__main__':
    main()
