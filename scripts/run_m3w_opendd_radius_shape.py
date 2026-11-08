"""Train fixed-radius trajectory shape models with pair-excluded producers."""
import argparse
import fcntl
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import sys
from time import monotonic

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_opendd_hindcast_residual as base

prior, frozen = base.prior, base.frozen
PUBLIC = base.PUBLIC.parent / 'opendd_radius_shape_v1'
PRIVATE = base.PRIVATE.parent / PUBLIC.name
INNER = PRIVATE.parent / 'opendd_velocity_calibration_v1'
SITES, SEEDS, STEPS, BATCH = base.SITES, base.SEEDS, base.STEPS, base.BATCH


def register():
    old = base.register()
    files = [Path(__file__).resolve(), PUBLIC/'protocol.md', base.PUBLIC/'registration.json',
             base.PUBLIC/'analysis.json', base.PUBLIC/'amplitude_audit.json',
             frozen.CALIBRATION/'analysis.json',
             ROOT/'src/world_model/m3w_radius_shape.py', ROOT/'src/world_model/m3w_hindcast_residual.py']
    donors = {}
    for pair in combinations(SITES, 2):
        for seed in SEEDS:
            name = f'{"+".join(pair)}_{seed}_self.pt'
            donors[name] = prior.sha(INNER/name)
    value = dict(previous_registration=old, files={str(p.relative_to(ROOT)):prior.sha(p) for p in files},
                 inner_checkpoint_sha256=donors, source='opendd_exposed_development_only',
                 held_locations=SITES, seeds=SEEDS, steps=STEPS, batch=BATCH, original_risk_budget=.02,
                 external_access=False, dronecrowd_confirmation_access=False)
    p = PUBLIC/'registration.json'
    prior.write_json(p, value, immutable=True)
    return prior.sha(p), donors


def load_producers(torch, model, donors):
    outer = frozen.load_networks(torch, model)
    reg = prior.sha(frozen.CALIBRATION/'registration.json')
    receipt = json.loads((frozen.CALIBRATION/'analysis.json').read_text())
    if receipt['registration'] != reg:
        raise ValueError('Unexpected producer analysis provenance')
    traces = {(r['held'], r['seed']): r for r in receipt['fresh_inner_fits']}
    inner = {}
    for pair in combinations(SITES, 2):
        held = '+'.join(pair)
        for seed in SEEDS:
            name = f'{held}_{seed}_self.pt'
            path = INNER/name
            if prior.sha(path) != donors[name]:
                raise ValueError('Registered inner producer changed')
            ck = torch.load(path, map_location='cpu', weights_only=True)
            expected = dict(registration=reg, held=held, train=[s for s in SITES if s not in pair],
                            seed=seed, arm='self')
            if ck['identity'] != expected or ck['step'] != 600 or traces[held, seed] != dict(**expected, trace=ck['trace']):
                raise ValueError('Producer did not exclude both required locations')
            net = model.network(113)
            net.load_state_dict(ck['model'])
            inner[pair, seed] = net.eval()
    return outer, inner


def site_data(site, torch, model, method):
    d = base.load_site(site)
    with np.load(prior.PRIVATE/f'{site}_cohort.npz', allow_pickle=False) as f:
        h, offsets = f['history'], f['offsets']
    x, _, _ = model.design(h, method.neighbors(h, offsets))
    if not np.array_equal(x[:, :17], d['x'].numpy()[:, :17]):
        raise ValueError('Peer input changed the own-history design')
    target = ((d['y']-d['cv'])@d['rot']) / d['scale'][:, None, None]
    return dict(**d, peer=torch.from_numpy(x), offsets=offsets,
                residual_target=torch.from_numpy(target.astype(np.float32)))


def candidate(d, producer, torch, model):
    with torch.inference_mode():
        direct = model.decode(producer(d['x'])).numpy().astype(float)
    direct = (direct*d['scale'][:, None, None]) @ d['rot'].transpose(0, 2, 1)
    inc, old = frozen.blend(d['cv'], direct, d['old'], d['legal'], .05)
    delta = inc-d['cv']
    r = np.linalg.norm(delta, axis=-1).max(1)
    pool = np.flatnonzero(r > 0)
    unit = np.divide(delta@d['rot'], r[:, None, None], out=np.zeros_like(delta), where=r[:, None, None]>0)
    def tensor(a):
        return torch.from_numpy(a.astype(np.float32))

    added = torch.column_stack((tensor(unit.reshape(len(r), -1)), tensor(np.log1p(r/d['scale']))))
    return dict(inc=inc, old=old, r=r, pool=pool, unit=tensor(unit),
                ratio=tensor(r/d['scale']),
                x=torch.cat((d['x'], added), 1), peer=torch.cat((d['peer'], added), 1))


def fit(sources, candidates, held, seed, arm, reg, torch, model, method, obj, expected):
    train = [s for s in SITES if s != held]
    cut = base.cuts_for([sources[s] for s in train])[0]
    donors = [candidates[s] for s in train]
    easy = {s: torch.from_numpy((sources[s]['cv_ade']>0) & (sources[s]['cv_ade']<=cut)) for s in train}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+1776)
    net = model.network(138)
    torch.nn.init.zeros_(net[-1].weight)
    torch.nn.init.zeros_(net[-1].bias)
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    ident = dict(registration=reg, held=held, seed=seed, arm=arm, train=train)
    path = PRIVATE/f'{held}_{seed}_{arm}.pt'
    if expected is not None and prior.sha(path) != expected:
        raise ValueError('Replay refuses a changed or missing checkpoint')
    step, trace = 0, []
    if path.exists():
        ck = torch.load(path, map_location='cpu', weights_only=True)
        if ck['identity'] != ident or ck['step'] not in range(100, STEPS+1, 100):
            raise ValueError('Checkpoint provenance mismatch')
        net.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        rng.bit_generator.state = ck['rng']
        step, trace = ck['step'], ck['trace']
    if expected is not None and step != STEPS:
        raise ValueError('Replay cannot train')
    for i in range(step, STEPS):
        draw = rng.integers(len(donors), size=BATCH)
        picks = [(j, rng.choice(d['pool'], size=int((draw==j).sum()))) for j,d in enumerate(donors)]

        def get(k):
            return torch.cat([donors[j][k][idx] for j,idx in picks])

        x = get('peer' if arm=='peer_guard' else 'x')
        unit, ratio = get('unit'), get('ratio')[:, None, None]
        y = torch.cat([sources[train[j]]['residual_target'][idx] for j,idx in picks])
        valid = torch.cat([sources[train[j]]['local'][3][idx] for j,idx in picks])
        e = torch.cat([easy[train[j]][idx] for j,idx in picks])
        pred = ratio*method.shape(net(x), unit)
        loss = obj.objective(pred, torch.zeros_like(pred), ratio*unit, y, valid, e,
                             'easy_guard' if arm=='peer_guard' else arm)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite shape loss: {ident}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (i+1)%100 == 0:
            trace.append(float(loss.detach()))
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=ident, model=net.state_dict(), optimizer=opt.state_dict(),
                step=i+1, trace=trace, rng=rng.bit_generator.state), tmp)
            tmp.replace(path)
            prior.write_json(PRIVATE/'heartbeat.json',
                dict(**ident, phase='train', step=i+1, loss=trace[-1], pid=os.getpid()))
    return net.eval(), dict(identity=ident, trace=trace, sha256=prior.sha(path)), STEPS-step


def pair_errors(d, paths):
    left, right = [], []
    for lo, hi in zip(d['offsets'][:-1], d['offsets'][1:], strict=True):
        i, j = np.triu_indices(hi-lo, k=1)
        left.extend(lo+i)
        right.extend(lo+j)
    i, j = np.asarray(left, dtype=int), np.asarray(right, dtype=int)
    both = d['valid'][i] & d['valid'][j]
    target = d['y'][i]-d['y'][j]
    # Origins cancel in predicted-minus-true relative vectors.
    errors = {name:float(np.linalg.norm((p[i]-p[j])-target, axis=-1)[both].mean())
              for name,p in paths.items()}
    return dict(pairs=len(i), observed_pair_steps=int(both.sum()), mean_relative_vector_error=errors)


def compare_peer(rows):
    draws = np.random.default_rng(817).integers(len(SITES), size=(3000,len(SITES)))
    x = np.asarray([np.mean([v['peer_vs_self']['lower_pp'] for v in rows[s]]) for s in SITES])
    pair = np.asarray([np.mean([v['pairs']['mean_relative_vector_error']['peer_guard']-
                                v['pairs']['mean_relative_vector_error']['easy_guard'] for v in rows[s]]) for s in SITES])
    return dict(lower_pp=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025,.975]).tolist(),
                relative_error_change=float(pair.mean()),
                relative_error_change_ci95=np.quantile(pair[draws].mean(1), [.025,.975]).tolist())


def run(reg, donors, replay):
    import torch
    from src.world_model import m3w_opendd_velocity_candidate as model
    from src.world_model import m3w_hindcast_residual as obj
    from src.world_model import m3w_radius_shape as method

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    t0 = monotonic()
    p = PUBLIC/'analysis.json'
    cp = PUBLIC/'checkpoint_freeze.json'
    expected = json.loads(cp.read_text()) if replay else None
    if replay and (not p.is_file() or expected['registration'] != reg):
        raise ValueError('Replay requires a completed registered readout')
    sources = {s:site_data(s,torch,model,method) for s in SITES}
    outer, inner = load_producers(torch, model, donors)
    nets, metadata, updates = {}, {}, 0
    for held in SITES:
        print(f'{"Verifying" if replay else "Fitting"} {held}', flush=True)
        for seed in SEEDS:
            c = {s:candidate(sources[s], inner[tuple(sorted((s,held))), seed], torch, model)
                 for s in SITES if s != held}
            for arm in method.ARMS:
                key = f'{held}/{seed}/{arm}'
                net, meta, n = fit(sources,c,held,seed,arm,reg,torch,model,method,obj,
                                   None if expected is None else expected['fits'][key]['sha256'])
                nets[held,seed,arm], metadata[key] = net, meta
                updates += n
    prior.write_json(cp, dict(registration=reg,fits=metadata), immutable=True)
    rows = {}
    for i, site in enumerate(SITES):
        d = sources[site]
        cv = d['cv']
        cuts = base.cuts_for([sources[s] for s in SITES if s != site])
        rows[site] = []
        for seed in SEEDS:
            c = candidate(d,outer[seed][i],torch,model)
            r = c['r']
            support = r>0
            paths = dict(cv=cv,incumbent=c['inc'],hindcast=cv+method.at_radius(c['old']-cv,r))
            for arm in method.ARMS:
                net = nets[site,seed,arm]
                with torch.inference_mode():
                    sh = method.shape(net(c['peer' if arm=='peer_guard' else 'x'][support]),c['unit'][support])
                world = np.zeros_like(cv)
                world[support] = sh.numpy().astype(float) @ d['rot'][support].transpose(0,2,1)
                paths[arm] = cv+method.at_radius(world,r)
            for name,pred in paths.items():
                if name != 'cv':
                    np.testing.assert_allclose(np.linalg.norm(pred-cv, axis=-1).max(1), r, rtol=1e-7, atol=1e-14)
            rows[site].append(dict(seed=seed,cuts=cuts.tolist(),arms=base.readout(d,paths,support,cuts),
                peer_vs_self=base.paired_summary(d,paths['peer_guard'],paths['easy_guard'],support),
                pairs=pair_errors(d,paths)))
    prior.write_json(p, dict(registration=reg, fits=len(nets), inner_producers=len(inner),
        outer_producers=sum(map(len,outer.values())), targets=sum(len(d['y']) for d in sources.values()),
        summary=base.summary(rows), by_location=rows, peer_vs_self=compare_peer(rows),
        exact_radius_matching=True, formal_confirmation=False, deployment=False,
        stage5c_executed=False, smc_enabled=False), immutable=True)
    receipt = PUBLIC/('replay_execution.json' if replay else 'execution.json')
    if not receipt.exists():
        prior.write_json(receipt, dict(registration=reg,analysis_sha256=prior.sha(p),
            updates_this_execution=updates,fresh_exact_replay=replay,pid=os.getpid(),torch_threads=4,workers=0,
            wall_seconds=round(monotonic()-t0,3)), immutable=True)
    prior.write_json(PRIVATE/'heartbeat.json', dict(phase='complete',pid=os.getpid(),updates=updates,replay=replay))
    print(prior.sha(p), flush=True)


if __name__=='__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('phase',choices=('register','run','replay'))
    phase = a.parse_args().phase
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reg, donors = register()
        if phase=='register':
            print(reg)
        else:
            if platform.system()!='Darwin' or platform.machine()!='arm64':
                raise RuntimeError('Use native arm64 .venv-pytorch on this host')
            run(reg,donors,phase=='replay')
