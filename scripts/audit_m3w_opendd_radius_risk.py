"""Attribute risk changes at identical recipients and CV departure."""
from collections import Counter
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_opendd_radius_shape as run

LIMITS = ('completion_all_risk_upper', 'completion_easy_risk_upper',
          'known_easy_net_degradation')


def transition(old, new):
    if any((old[k] is None) != (new[k] is None) for k in LIMITS):
        raise ValueError('Matched forecast unexpectedly changed reference support')
    if old['supported'] and not new['supported']:
        return 'lost_pass'
    if not old['supported'] and new['supported']:
        return 'gained_pass'
    if old['supported']:
        return 'both_pass'
    return 'both_undefined' if any(old[k] is None for k in LIMITS) else 'both_defined_fail'


def compare(d, old, new, cuts):
    from src.evaluation.m3w_selection_forecast_readout import errors, policy

    cv, y, valid, rec = (d[k] for k in ('cv', 'y', 'valid', 'recordings'))
    ro = np.linalg.norm(old-cv, axis=-1).max(1)
    rn = np.linalg.norm(new-cv, axis=-1).max(1)
    np.testing.assert_array_equal(ro>0, rn>0)
    np.testing.assert_allclose(ro, rn, rtol=1e-7, atol=1e-14)
    a, b = (policy(p, cv, cv, y, valid, rec, easy_cut=cuts[0], hard_cut=cuts[1])['recordings']
            for p in (old,new))
    ce, _ = errors(cv,y,valid)
    known = valid.any(1)
    easy = known & (ce>0) & (ce<=cuts[0])
    result = {}
    for recording in a:
        at = rec==recording
        mass = float(ce[at & easy & (ro>0)].sum())
        u = float(ro[at & ~known].sum())
        result[recording] = dict(transition=transition(a[recording],b[recording]),
            selected_easy_mass=mass, same_unknown_debit=u,
            incumbent={k:a[recording][k] for k in LIMITS},
            new={k:b[recording][k] for k in LIMITS})
    return result, sum(r['supported'] for r in a.values()), sum(r['supported'] for r in b.values())


def main():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Use native arm64 .venv-pytorch on this host')
    import torch
    from src.world_model import m3w_opendd_velocity_candidate as model
    from src.world_model import m3w_radius_shape as method

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg, donors = run.register()
    p = run.PUBLIC/'analysis.json'
    digest = run.prior.sha(p)
    evidence = json.loads(p.read_text())
    replay = json.loads((run.PUBLIC/'replay_execution.json').read_text())
    freeze = json.loads((run.PUBLIC/'checkpoint_freeze.json').read_text())
    if (evidence['registration'] != reg or freeze['registration'] != reg or
        not replay['fresh_exact_replay'] or replay['analysis_sha256'] != digest):
        raise ValueError('Risk audit requires the exact replayed experiment')
    outer = run.frozen.load_networks(torch, model)
    rows = {}
    for i,site in enumerate(run.SITES):
        d = run.site_data(site,torch,model,method)
        rows[site] = []
        for view in evidence['by_location'][site]:
            seed = view['seed']
            c = run.candidate(d,outer[seed][i],torch,model)
            support = c['r']>0
            p = run.PRIVATE/f'{site}_{seed}_easy_guard.pt'
            if run.prior.sha(p) != freeze['fits'][f'{site}/{seed}/easy_guard']['sha256']:
                raise ValueError('Model checkpoint changed')
            net = model.network(138)
            net.load_state_dict(torch.load(p,weights_only=True,map_location='cpu')['model'])
            with torch.inference_mode():
                sh = method.shape(net.eval()(c['x'][support]),c['unit'][support])
            world = np.zeros_like(d['cv'])
            world[support] = sh.numpy().astype(float)@d['rot'][support].transpose(0,2,1)
            new = d['cv']+method.at_radius(world,c['r'])
            records, old_n, new_n = compare(d,c['inc'],new,view['cuts'])
            if old_n != view['arms']['incumbent']['passed'] or new_n != view['arms']['easy_guard']['passed']:
                raise ValueError('Risk attribution does not match frozen pass counts')
            rows[site].append(dict(seed=seed,records=records))
    flat = [r for views in rows.values() for v in views for r in v['records'].values()]
    groups = {key:[r for r in flat if r['transition']==key] for key in
              ('lost_pass','gained_pass','both_pass','both_defined_fail','both_undefined')}
    flips = {key:Counter(k for r in groups[key] for k in LIMITS if r[side][k]>.02)
             for key,side in (('lost_pass','new'),('gained_pass','incumbent'))}
    medians = {k:float(np.median([r['selected_easy_mass'] for r in seq]))
               for k,seq in groups.items() if seq}
    run.prior.write_json(run.PUBLIC/'risk_attribution.json', dict(original_analysis_sha256=digest,
        posthoc_diagnostic=True, decisions_changed=False, transitions={k:len(v) for k,v in groups.items()},
        flipped_risk_components=flips, median_selected_easy_mass=medians, by_location=rows), immutable=True)


if __name__=='__main__':
    main()
