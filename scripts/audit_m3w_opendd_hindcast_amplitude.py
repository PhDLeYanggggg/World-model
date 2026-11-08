"""Post-hoc matched-displacement audit; no fitting or policy selection."""
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_opendd_hindcast_residual as run


def radii(paths, cv):
    return {name: np.linalg.norm(p-cv, axis=-1).max(1) for name, p in paths.items()}


def constrain(paths, cv, mode):
    r = radii(paths, cv)
    if mode == 'equal_displacement':
        cap = np.minimum.reduce(list(r.values()))
    elif mode == 'incumbent_cap':
        cap = r['incumbent']
    else:
        raise ValueError(mode)
    out = {}
    for name, p in paths.items():
        factor = np.divide(cap, r[name], out=np.ones_like(cap), where=r[name]>0).clip(0., 1.)
        out[name] = cv+factor[:, None, None]*(p-cv)
    if mode == 'equal_displacement':
        for v in radii(out, cv).values():
            np.testing.assert_allclose(v, cap, rtol=1e-7, atol=1e-14)
    return out


def main():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Use native arm64 .venv-pytorch on this host')
    import torch
    from src.world_model import m3w_opendd_velocity_candidate as model

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg = run.register()
    p = run.PUBLIC / 'analysis.json'
    result = json.loads(p.read_text())
    digest = run.prior.sha(p)
    freeze = json.loads((run.PUBLIC / 'checkpoint_freeze.json').read_text())
    replay = json.loads((run.PUBLIC / 'replay_execution.json').read_text())
    if (result['registration'] != reg or freeze['registration'] != reg or
        replay['analysis_sha256'] != digest or not replay['fresh_exact_replay']):
        raise ValueError('The initial experiment must match its frozen replay')
    incumbent = run.frozen.load_networks(torch, model)
    studies = {m: {} for m in ('incumbent_cap', 'equal_displacement')}
    moves = {}
    with torch.inference_mode():
        for i, site in enumerate(run.SITES):
            d = run.load_site(site)
            cv = d['cv']
            moves[site] = []
            for study in studies.values():
                study[site] = []
            for view in result['by_location'][site]:
                seed, cuts = view['seed'], view['cuts']
                direct = model.decode(incumbent[seed][i](d['x'])).numpy().astype(float)
                direct = (direct*d['scale'][:, None, None]) @ d['rot'].transpose(0, 2, 1)
                inc, old = run.frozen.blend(cv, direct, d['old'], d['legal'], .05)
                active = (old != cv).any(axis=(1, 2))
                path = run.PRIVATE / f'{site}_{seed}_easy_guard.pt'
                if run.prior.sha(path) != freeze['fits'][f'{site}/{seed}/easy_guard']['checkpoint_sha256']:
                    raise ValueError('Unverified safety checkpoint')
                ck = torch.load(path, weights_only=True, map_location='cpu')
                net = model.network(113)
                net.load_state_dict(ck['model'])
                delta = (.25*.05)*model.decode(net.eval()(d['x'])).numpy().astype(float)
                world = (delta*d['scale'][:, None, None]) @ d['rot'].transpose(0, 2, 1)
                paths = dict(hindcast=old, incumbent=inc, easy_guard=old+np.where(active[:, None, None], world, 0.))
                source = run.paired_summary(d, paths['easy_guard'], old, active)
                if source != view['arms']['easy_guard']['comparisons']['hindcast']:
                    raise ValueError('Post-hoc audit does not reproduce the frozen primary readout')
                r = radii(paths, cv)
                ratio = r['easy_guard'][active] / r['incumbent'][active]
                moves[site].append(dict(seed=seed, rows=int(active.sum()), over_incumbent=int((ratio>1.).sum()),
                                        ratio_q25_median_q75=np.quantile(ratio, [.25, .5, .75]).tolist()))
                for mode, study in studies.items():
                    candidate = constrain(paths, cv, mode)
                    study[site].append(dict(seed=seed, arms=run.readout(d, dict(cv=cv, **candidate), active, cuts)))
    run.prior.write_json(run.PUBLIC / 'amplitude_audit.json',
        dict(posthoc_diagnostic=True, original_analysis_sha256=digest, movement=moves,
             studies={m: dict(summary=run.summary(x), by_location=x) for m, x in studies.items()},
             decision_changed=False, fitting_updates=0), immutable=True)


if __name__ == '__main__':
    main()
