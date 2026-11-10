"""Fit and score a pinned Social-STGCNN baseline on the frozen vehicle cohort."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
from time import monotonic

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_vehicle_eqmotion as match
from scripts.run_m3w_opendd_recipient_transport import require_published
from src.world_model import m3w_vehicle_stgcnn as model

vehicle, np, torch, prior = match.vehicle, match.np, match.torch, match.prior
SITES, SEEDS, read = match.SITES, match.SEEDS, match.read
OUT = match.OUT.parent/'opendd_vehicle_stgcnn_v1'
LOCAL = match.LOCAL.parent/OUT.name


def register():
    match_key, key, incumbent, geometry = match.register()
    model.author()
    previous = match.OUT/'readout.json'
    v = read(previous)
    if v['binding']['freeze'] != prior.sha(match.OUT/'freeze.json') or v['external_outcomes_opened']:
        raise ValueError('Require the original frozen EqMotion comparison')
    r = OUT/'registration.json'
    inputs = (Path(__file__), Path(model.__file__), Path(match.__file__),
              Path(match.model.__file__), Path(match.model.eq.__file__), OUT/'protocol.md', previous)
    prior.write_json(r, dict(match_registration=match_key,
        files={str(p.relative_to(prior.ROOT)): prior.sha(p) for p in inputs},
        author_revision=model.UPSTREAM, author_digests=model.DIGESTS, architecture=model.ARCH,
        arms=model.ARMS, sites=SITES, seeds=SEEDS, batch=vehicle.BATCH, steps=vehicle.STEPS,
        checkpoint_every=vehicle.SAVE, threads=4, workers=0, original_2pct=True, external_outcomes_opened=False),
        immutable=True)
    return prior.sha(r), key, incumbent, geometry


def beat(phase, **kw):
    msg = dict(phase=phase, pid=os.getpid(), **kw)
    p = LOCAL/'heartbeat.pending'
    prior.write_json(p, msg)
    p.replace(LOCAL/'heartbeat.json')
    print(msg, flush=True)


def fit(key, data, site, seed, arm):
    identity = match.identity(key, site, seed, arm)
    path = LOCAL/f'{site}_{seed}_{arm}.pt'
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, SITES.index(site)])
    net = model.Forecast(arm)
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    start, begin = 0, monotonic()
    if path.exists():
        z = torch.load(path, map_location='cpu', weights_only=True)
        if z['identity'] != identity or not 0 <= z['step'] <= vehicle.STEPS:
            raise ValueError('Cannot resume a different fit')
        net.load_state_dict(z['weights'])
        opt.load_state_dict(z['optim'])
        rng.bit_generator.state, start = z['rng'], z['step']
    donors = identity['donors']
    for step in range(start+1, vehicle.STEPS+1):
        pick = rng.integers(len(donors), size=vehicle.BATCH)
        pieces = []
        for i, s in enumerate(donors):
            x = data[s]['eq']
            row = rng.integers(len(x[0]), size=int((pick == i).sum()))
            pieces.append(tuple(z[row] for z in x))
        batch = tuple(torch.cat(z) for z in zip(*pieces, strict=True))
        opt.zero_grad()
        loss = match.model.objective(net, batch)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite Social-STGCNN training loss')
        loss.backward()
        opt.step()
        if step % vehicle.SAVE == 0:
            temp = path.with_suffix('.pending')
            torch.save(dict(identity=identity, step=step, weights=net.state_dict(),
                            optim=opt.state_dict(), rng=rng.bit_generator.state), temp)
            temp.replace(path)
            beat('fit', site=site, seed=seed, arm=arm, step=step, loss=float(loss.detach()),
                 seconds=round(monotonic()-begin, 2))
    return dict(identity=identity, sha256=prior.sha(path), step=vehicle.STEPS)


@torch.inference_mode()
def predict(key, d, site, seed, arm, ref):
    path = LOCAL/f'{site}_{seed}_{arm}.pt'
    if prior.sha(path) != ref['sha256'] or ref['identity'] != match.identity(key, site, seed, arm):
        raise ValueError('Checkpoint identity or digest changed')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['identity'] != ref['identity'] or state['step'] != vehicle.STEPS:
        raise ValueError('Require the fixed final checkpoint')
    net = model.Forecast(arm).eval()
    net.load_state_dict(state['weights'])
    return torch.cat([match.model.bounded(net(x, p, m), cv, scale) for x, p, m, cv, scale in
                      zip(*(v.split(128) for v in d['eq'][:5]), strict=True)]).numpy().astype(float)


def score(key, data, refs):
    p = OUT/'freeze.json'
    require_published(p)
    f = read(p)
    expected = {f'{s}_{seed}_{a}' for s in SITES for seed in SEEDS for a in model.ARMS}
    if f['registration'] != key or set(f['checkpoints']) != expected or f['external_outcomes_opened']:
        raise ValueError('Require the complete training freeze')
    ref = read(match.OUT/'readout.json')
    binding = dict(freeze=prior.sha(p), eqmotion=prior.sha(match.OUT/'readout.json'))
    easy = {}
    for s, d in data.items():
        mask = d['valid'].all(1)
        val = np.linalg.norm(d['cv'][mask]-d['future'][mask], axis=-1).mean(1)
        easy[s] = val[val > 0.]
    rows = {}
    for s, d in data.items():
        cut = np.quantile(np.concatenate([easy[k] for k in SITES if k != s]), [.25, .75])
        cv, ols, y, valid, weight = (d[k] for k in ('cv', 'ols4', 'future', 'valid', 'weight'))
        mass = float(weight @ np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
        for seed in SEEDS:
            cell = f'{s}_{seed}'
            dest = LOCAL/f'{cell}_readout.json'
            if not dest.exists():
                pred = {a: predict(key, d, s, seed, a, f['checkpoints'][f'{cell}_{a}']) for a in model.ARMS}
                pred.update(cv=cv, ols4=ols, completion=match.predict_incumbent(s, seed, d, refs))
                scores = {a: vehicle.metric(v, cv, ols, d, cut) for a, v in pred.items()}
                previous = ref['by_site_seed'][cell]['metrics']
                for a in ('cv', 'ols4', 'completion'):
                    np.testing.assert_allclose(scores[a]['cv_gain_known_lower_pp'],
                                               previous[a]['cv_gain_known_lower_pp'], atol=1e-9, rtol=0.)
                def effect(a, b):
                    return [float(100.*weight @ z.sum(1)/mass) for z in vehicle.paired(pred[b], pred[a], y, valid)]
                pair = dict(social_peers_minus_ego=effect(*model.ARMS))
                pair.update({a+'_minus_incumbent': effect(a, 'completion') for a in model.ARMS})
                prior.write_json(dest, dict(binding=binding, metrics=scores, paired=pair), immutable=True)
            rows[cell] = read(dest)
            if rows[cell]['binding'] != binding:
                raise ValueError('Cached score has a different source freeze')
            beat('score', site=s, seed=seed, completed=len(rows))
    draws = np.random.default_rng(3183).integers(len(SITES), size=(3000, len(SITES)))
    def ci(get):
        x = np.array([np.mean([get(rows[f'{s}_{seed}']) for seed in SEEDS]) for s in SITES])
        return dict(mean=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist(), sites=x.tolist())
    metrics = {}
    for a in (*model.ARMS, 'completion', 'cv', 'ols4'):
        m = {k: ci(lambda r: r['metrics'][a][k]) for k in
             ('known_ade_native', 'fde_available_native', 'easy_gain_percent', 'p95_ade_native')}
        m['risk_passes'] = sum(r['metrics'][a]['risk_passes'] for r in rows.values())
        for k in ('cv', 'ols'):
            m[k+'_gain_lower_pp'] = ci(lambda r: r['metrics'][a][k+'_gain_known_lower_pp'][1])
        metrics[a] = m
    paired = {k: {m: ci(lambda r: r['paired'][k][i]) for i, m in enumerate(('observed', 'lower'))}
              for k in next(iter(rows.values()))['paired']}
    prior.write_json(OUT/'readout.json', dict(binding=binding, metrics=metrics, paired=paired, by_site_seed=rows,
        result_source='fresh_run', original_2pct=True, external_outcomes_opened=False,
        independent_confirmation=False, deployment_changed=False), immutable=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'score'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohort, incumbent, geometry = register()
        data = match.load(cohort, geometry)
        if args.phase == 'fit':
            trained = {f'{s}_{seed}_{a}': fit(key, data, s, seed, a)
                       for s in SITES for seed in SEEDS for a in model.ARMS}
            prior.write_json(OUT/'freeze.json', dict(registration=key, checkpoints=trained,
                optimizer_updates=len(trained)*vehicle.STEPS, external_outcomes_opened=False), immutable=True)
        else:
            score(key, data, incumbent)


if __name__ == '__main__':
    main()
