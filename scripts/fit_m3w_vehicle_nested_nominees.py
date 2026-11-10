"""Fit the strictly source-excluded teachers for nominee-level ranking."""

from itertools import combinations
import fcntl
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_opportunity as screen

action, ctx, np, torch, base = screen.action, screen.ctx, screen.np, screen.torch, screen.base
old, sha, read, write = screen.old, screen.sha, screen.read, screen.write
OUT = screen.OUT.parent/'opendd_vehicle_nominee_rank_v1'
LOCAL = screen.LOCAL.parent/OUT.name


def manifest():
    k, source, forecasts, outer, ego, cap = screen.register()
    frozen = read(screen.OUT/'freeze.json')
    if frozen['registration'] != k or not frozen['before_held_scoring'] or len(frozen['cells']) != 63:
        raise ValueError('Require previously frozen held nominees')
    files = (Path(__file__), Path(action.__file__), Path(action.model.__file__), Path(old.__file__),
             Path(old.motion.__file__), screen.OUT/'freeze.json', OUT/'protocol.md')
    p = OUT/'teacher_registration.json'
    write(p, dict(inputs={str(f.resolve().relative_to(base.prior.ROOT)): sha(f) for f in files},
        seeds=base.SEEDS, forecaster_steps=base.parent.STEPS, ego_steps=action.STEPS,
        donor_exclusions=(4, 3), threads=4, workers=0, external_outcomes_read=False), immutable=True)
    return sha(p), source, forecasts, outer, ego, cap, frozen


def tag(sites, seed):
    return '_'.join(sorted(sites))+f'_{seed}'


def data_slice(d, y):
    return dict(history=d['history'], query_counts=d['counts'], recording=y['recording'],
                tensors=(torch.from_numpy(d['x'][:, :17]), torch.from_numpy(d['cv'])),
                rotation=d['rotation'], scale=d['scale'])


def forecast_targets(d):
    y = action.labels(d['file'])
    x, cv = data_slice(d, y)['tensors']
    target = (y['future'] @ d['rotation']) / d['scale'][:, None, None]
    return x, cv, torch.from_numpy(target.astype(np.float32)), torch.from_numpy(y['valid'])


def anchor_examples(d, site, reject, seed, forecast):
    if site not in reject or len(set(reject)) != len(reject):
        raise ValueError('The proposal forecaster must exclude this source')
    y = action.labels(d['file'])
    m = action.study.menu(data_slice(d, y), forecast)
    rng = np.random.default_rng([seed, *(base.SITES.index(s) for s in sorted(reject)), base.SITES.index(site)])
    for lo, hi in m['slots'].values():
        q = slice(lo, hi)
        cv = m['cv'][q]
        legal = m['peak'][q] > 0.
        rows = np.flatnonzero(legal.any(1))
        for _ in range(action.DRAWS):
            a = rng.choice(rows, 2, replace=False)
            f = [rng.choice(np.flatnonzero(legal[i])) for i in a]
            anchor = cv.copy()
            anchor[a] = m['paths'][lo+a, f]
            x, angles = action.inputs(cv, anchor, a, d, q)
            yield m, q, a, anchor, x, angles, y


def action_batches(d, site, ex, seed, ref):
    exclude = sorted([*ex, site])
    if len(exclude) != 4 or len(set(exclude)) != 4:
        raise ValueError('Require three excluded locations plus the inner donor')
    forecast = base.load_net(LOCAL/f'{tag(exclude, seed)}_forecast.pt', ref,
                             old.motion.motion.network(17), exclude)
    x, labels = [], []
    for m, q, a, anchor, feats, angles, y in anchor_examples(d, site, exclude, seed, forecast):
        x.append(feats.reshape(-1, action.model.WIDTH))
        labels.append(action.model.targets(anchor, a, angles, y['future'][q], y['valid'][q]).reshape(-1, 3))
    return torch.from_numpy(np.concatenate(x)), torch.from_numpy(np.concatenate(labels))


def fitted(identity, batches, loss, net, steps):
    saved = old.LOCAL
    try:
        old.LOCAL = LOCAL
        return old.fit(identity, batches, loss, net, steps, 128)[1]
    finally:
        old.LOCAL = saved


def stages(key, data):
    heads = {}
    for order, suffix, steps in ((4, 'forecast', base.parent.STEPS), (3, 'ego', action.STEPS)):
        out = {}
        for excluded in combinations(base.SITES, order):
            donors = [s for s in base.SITES if s not in excluded]
            for seed in base.SEEDS:
                name = f'{tag(excluded, seed)}_{suffix}'
                identity = dict(registration=key, name=name, seed=seed, excluded=list(excluded), fit_sites=donors)
                receipt = LOCAL/f'{name}.json'
                if not receipt.exists():
                    if order == 4:
                        train = [forecast_targets(data[s]) for s in donors]

                        def loss(net, b):
                            return old.motion.loss(old.motion.predict(net, b[0], b[1]), *b[1:], completion=True)

                        network = lambda: old.motion.motion.network(17)
                    else:
                        train = [action_batches(data[s], s, excluded, seed, heads['forecast'][
                            tag((*excluded, s), seed)]) for s in donors]
                        loss, network = action.model.loss, lambda: action.model.AngleValue(False)
                    torch.manual_seed(seed)
                    write(receipt, fitted(identity, train, loss, network(), steps), immutable=True)
                r = read(receipt)
                if r['identity'] != identity or r['step'] != steps or sha(LOCAL/f'{name}.pt') != r['sha256']:
                    raise ValueError('Changed nested checkpoint')
                out[tag(excluded, seed)] = r
                write(LOCAL/'teacher_heartbeat.json', dict(pid=os.getpid(), stage=suffix, completed=len(out), total=105))
                if len(out) % 7 == 0:
                    print(suffix, len(out), '/ 105', flush=True)
        heads[suffix] = out
    write(OUT/'teacher_freeze.json', dict(registration=key, checkpoints=heads,
        updates={a: len(z)*next(iter(z.values()))['step'] for a, z in heads.items()},
        external_outcomes_read=False), immutable=True)


if __name__ == '__main__':
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use native arm64 .venv-pytorch')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL/'teacher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, source, *_ = manifest()
        stages(key, ctx.load_features(source))
