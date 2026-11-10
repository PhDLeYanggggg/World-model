"""Fit and evaluate observation-aware ranking on fixed OpenDD nominations."""

import argparse
from collections import defaultdict
import fcntl
import os
from pathlib import Path
import platform
import sys

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use native arm64 .venv-pytorch')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import average_precision_score, roc_auc_score
from scripts import run_m3w_opendd_realized_swap_value as old
from src.world_model import m3w_swap_observation_support as model

np, torch, base = old.np, old.torch, old.base
read, write, sha = old.read, old.write, old.sha
OUT = old.OUT.parent/'opendd_swap_support_v1'
LOCAL = old.LOCAL.parent/OUT.name
ARMS = ('core', 'footprint')
STEPS, SAVE, BATCH = 300, 50, 32


def beat(phase, **info):
    write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, **info))
    print(phase, info, flush=True)


def register():
    parent_key, source, original, parent, _ = old.register()
    f, r, a = (old.OUT/k for k in ('freeze.json', 'readout.json', 'value_audit.json'))
    frozen, readout, audit = map(read, (f, r, a))
    if (frozen['key'] != parent_key or readout['freeze'] != sha(f) or audit['fit_freeze'] != sha(f)
            or audit['readout'] != sha(r) or not readout['original_2pct'] or readout['external_outcomes_read']):
        raise ValueError('Require the archived fixed-nomination experiment')
    path = OUT/'registration.json'
    files = (Path(__file__), Path(model.__file__), OUT/'protocol.md', f, r, a, old.prior.OUT/'freeze.json')
    write(path, dict(files={str(p.relative_to(base.prior.ROOT)): sha(p) for p in files},
        seeds=base.SEEDS, sites=base.SITES, steps=STEPS, threads=4, workers=0, checkpoint_every=SAVE,
        batch_per_donor=BATCH, arms=ARMS, original_2pct=True, external_outcomes_read=False), immutable=True)
    return sha(path), source, original, parent, frozen, audit


def packet(r):
    return old.prior.prior.packet(r)


def candidate(f, cv, anchor, q, a, b, new, x):
    data, scale, g = old.model.geometry(cv[q], anchor[q], a, b, new, x)
    footprint = model.footprint(f['history'][q], f['offsets'][q], cv[q], a, b)
    return np.column_stack((data, footprint)), scale, g


def donors(held, seed, original, features):
    data, proof = {}, {}
    for name, c in original['cells'].items():
        if c['seed'] != seed or held not in c['packets']:
            continue
        old.old.check_cell(c)
        s, = set(c['packets'])-{held}
        if s in data:
            raise ValueError('Duplicate donor')
        f, orig = features[s], c['packets'][s]
        with packet(orig) as p, np.load(f['file'], allow_pickle=False) as d:
            cv, anchor, bounds = old.old.geometry(p, f)
            y, v = d['future'], d['valid']
            rows = []
            for q, a, b, new, x in old.menus(p, f, cv, anchor, bounds):
                if len(x):
                    inp, _, g = candidate(f, cv, anchor, q, a, b, new, x)
                    rows.append((inp, model.targets(y[q], v[q], a, b, g)))
        data[s] = tuple(torch.from_numpy(np.concatenate(z)) for z in zip(*rows, strict=True))
        proof[s] = dict(cell=name, packet=orig['sha256'], cohort=sha(f['file']), candidates=len(data[s][0]))
    if set(data) != set(base.SITES)-{held}:
        raise ValueError('The held source cannot supply training labels')
    return [data[s] for s in sorted(data)], proof


def fit(identity, data):
    held, seed, arm = (identity[k] for k in ('held', 'seed', 'arm'))
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, base.SITES.index(held)])
    net = model.network(data[0][0].shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    p = LOCAL/f'{held}_{seed}_{arm}.pt'
    start = 0
    if p.exists():
        z = torch.load(p, map_location='cpu', weights_only=True)
        if z['identity'] != identity or not 0 <= z['step'] <= STEPS:
            raise ValueError('Changed training identity')
        start = z['step']
        net.load_state_dict(z['weights'])
        opt.load_state_dict(z['optimizer'])
        rng.bit_generator.state = z['rng']
    for step in range(start+1, STEPS+1):
        parts = [(x[i], y[i]) for x, y in data for i in (rng.integers(len(y), size=BATCH),)]
        x, y = (torch.cat(a) for a in zip(*parts, strict=True))
        if arm == 'core':
            x[:, -9:] = 0.
        loss = model.loss(net(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            temp = p.with_suffix('.pending')
            torch.save(dict(identity=identity, step=step, weights=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state), temp)
            temp.replace(p)
            beat('fit', held=held, seed=seed, arm=arm, step=step, loss=float(loss.detach()))
    return net.eval(), dict(file=p.name, sha256=sha(p), step=STEPS)


@torch.inference_mode()
def propose(p, f, actions, heads):
    cv, anchor, bounds = old.old.geometry(p, f)
    rows = []
    for i, (q, a, b, new, x) in enumerate(old.menus(p, f, cv, anchor, bounds)):
        j = int(actions[i])
        if j < 0:
            if len(x):
                raise ValueError('Missing frozen nomination')
            continue
        s = slice(j, j+1)
        inp, _, _ = candidate(f, cv, anchor, q, a[s], b[s], new[s], x[s])
        rows.append(inp[0])
    x = torch.from_numpy(np.stack(rows))
    result = {'footprint': heads['footprint'](x).sigmoid().numpy()}
    x[:, -9:] = 0.
    result['core'] = heads['core'](x).sigmoid().numpy()
    return result


def checked_local(c, root, expected):
    p = root/c['file']
    if c['original'] != expected or sha(p) != c['sha256']:
        raise ValueError('Wrong frozen nomination')
    return p


def freeze(key, source, original, previous):
    features, cells = old.old.prior.load_features(source), {}
    for held in base.SITES:
        for seed in base.SEEDS:
            tag = f'{held}_{seed}'
            path = LOCAL/f'{tag}.json'
            parent = previous['cells'][tag]
            if held in parent['fit']:
                raise ValueError('Frozen donor head used held outcomes')
            if not path.exists():
                train, proof = donors(held, seed, original, features)
                heads, ckpts, packets = {}, {}, {}
                for arm in ARMS:
                    heads[arm], ckpts[arm] = fit(dict(key=key, held=held, seed=seed, arm=arm, donors=proof), train)
                for name, c in original['cells'].items():
                    if c['seed'] != seed or held not in c['packets']:
                        continue
                    r, ref = c['packets'][held], parent['packets'][name]
                    p = checked_local(ref, old.LOCAL, r['sha256'])
                    with packet(r) as data, np.load(p, allow_pickle=False) as z:
                        arrays = propose(data, features[held], z['actions'], heads)
                    dest = LOCAL/f'{tag}_{name}.npz'
                    packets[name] = dict(file=dest.name, sha256=base.parent.indices.save_npz(dest, arrays),
                                         original=r['sha256'], parent=ref['sha256'])
                write(path, dict(key=key, fit=proof, ckpts=ckpts, packets=packets), immutable=True)
            c = read(path)
            if (c['key'] != key or held in c['fit'] or len(c['packets']) != 6 or
                    any(sha(LOCAL/r['file']) != r['sha256'] for r in (*c['ckpts'].values(), *c['packets'].values()))):
                raise ValueError('Changed registered scores or fitting roles')
            cells[tag] = c
            beat('freeze', completed=len(cells), total=21)
    write(OUT/'freeze.json', dict(key=key, cells=cells, scores_frozen=True, external_outcomes_read=False), immutable=True)


def evaluate(p, f, d, prev, baseline, new):
    cv, anchor, bounds = old.old.geometry(p, f)
    future, valid = d['future'], d['valid']
    if not np.array_equal(prev['actions'], baseline['actions']):
        raise ValueError('Old comparators differ in their nominations')
    targets, signs = [], []
    for i, (q, a, b, path, x) in enumerate(old.menus(p, f, cv, anchor, bounds)):
        j = int(prev['actions'][i])
        if j < 0:
            continue
        s = slice(j, j+1)
        _, _, g = candidate(f, cv, anchor, q, a[s], b[s], path[s], x[s])
        targets.append(model.targets(future[q], valid[q], a[s], b[s], g)[0])
        direct = old.model.targets(future[q], valid[q], a[s], b[s], g)[0]
        signs.append(direct[[0, 2]] > direct[[1, 3]])
    truth = np.stack(targets)
    ok = prev['actions'] >= 0
    for a in ARMS:
        if new[a].shape != (len(truth), 2, 4):
            raise ValueError('Prediction roster is incomplete')
    q = prev['prediction'][ok]
    fused = (q[:, [0, 2]]-q[:, [1, 3]])*prev['scale'][ok]
    return dict(truth=truth, sign=np.stack(signs), scale=prev['scale'][ok], fused=fused, raw_gain=baseline['raw_gain'][ok],
                **{a: new[a] for a in ARMS})


def score_group(group):
    x = {k: np.concatenate([g[k] for g in group]) for k in group[0]}
    result = {}
    for j, name in enumerate(('trajectory', 'relative')):
        label = x['sign'][:, j]
        if not label.any() or label.all():
            raise ValueError('A required location/seed cannot estimate ranking')
        p = {'v196_fused': x['fused'][:, j]}
        if j == 0:
            p['v194_gain'] = x['raw_gain']
        for k in ARMS:
            p[k+'_sign'] = x[k][:, j, 3]
            p[k+'_lower'] = model.lower(x[k][:, j])*x['scale'][:, j]
        result[name] = dict(views=len(label), positive=int(label.sum()),
            auc={k: float(roc_auc_score(label, v)) for k, v in p.items()},
            average_precision={k: float(average_precision_score(label, v)) for k, v in p.items()},
            support_mae={k: float(np.abs(x[k][:, j, 0]-x['truth'][:, j, 0]).mean()) for k in ARMS})
    return result


def summary(rows):
    groups = {s: [v for (site, _), v in rows.items() if site == s] for s in base.SITES}
    draw = np.random.default_rng(210).integers(7, size=(3000, 7))
    paired, means = {}, {}
    for task in ('trajectory', 'relative'):
        keys = rows[next(iter(rows))][task]
        means[task] = {m: {k: float(np.mean([v[task][m][k] for v in rows.values()])) for k in keys[m]}
                       for m in ('auc', 'average_precision', 'support_mae')}
        refs = [('footprint_sign', 'v196_fused'), ('footprint_sign', 'core_sign'), ('footprint_lower', 'v196_fused')]
        if task == 'trajectory':
            refs.append(('footprint_sign', 'v194_gain'))
        for a, b in refs:
            delta = np.array([np.mean([v[task]['auc'][a]-v[task]['auc'][b] for v in r]) for r in groups.values()])
            paired[f'{task}_{a}_minus_{b}'] = dict(point=float(delta.mean()),
                ci95=np.quantile(delta[draw].mean(1), [.025, .975]).tolist(),
                by_site=dict(zip(groups, delta.tolist(), strict=True)))
    gate = paired['relative_footprint_sign_minus_v196_fused']['ci95'][0] > 0. and (
        paired['trajectory_footprint_sign_minus_v194_gain']['point'] >= 0.)
    return dict(means=means, paired=paired, allocation_screen=gate, views=sum(v['trajectory']['views'] for v in rows.values()))


def score(key, source, original, parent, previous, audit):
    freeze = OUT/'freeze.json'
    old.old.require_published(freeze)
    frozen = read(freeze)
    if frozen['key'] != key or not frozen['scores_frozen'] or frozen['external_outcomes_read']:
        raise ValueError('Require the registered scores')
    features, data = old.old.prior.load_features(source), base.parent.load(source)
    groups = defaultdict(list)
    for name, c in original['cells'].items():
        old.old.check_cell(c)
        for held, orig in c['packets'].items():
            seed = c['seed']
            tag = f'{held}_{seed}'
            a, b, e = (m['cells'][tag]['packets'][name] for m in (frozen, previous, parent))
            if a['parent'] != b['sha256'] or b['nomination'] != e['sha256'] or held in frozen['cells'][tag]['fit']:
                raise ValueError('A prediction does not bind its held nominations')
            x, y, z = (checked_local(r, p, orig['sha256']) for r, p in ((a, LOCAL), (b, old.LOCAL), (e, old.prior.LOCAL)))
            local = LOCAL/f'{tag}_{name}_score.npz'
            receipt = local.with_suffix('.json')
            ident = dict(key=key, forecast=a['sha256'], old=b['sha256'], nomination=e['sha256'])
            if not receipt.exists():
                with packet(orig) as p, np.load(x, allow_pickle=False) as n, np.load(y, allow_pickle=False) as prev, np.load(z, allow_pickle=False) as v:
                    val = evaluate(p, features[held], data[held], prev, v, n)
                h = base.parent.indices.save_npz(local, val)
                write(receipt, dict(identity=ident, sha256=h), immutable=True)
            r = read(receipt)
            if r['identity'] != ident or r['sha256'] != sha(local):
                raise ValueError('Changed cached ranking evaluation')
            with np.load(local, allow_pickle=False) as f:
                groups[held, seed].append({k: f[k] for k in f.files})
        beat('score', completed=sum(map(len, groups.values())), total=126)
    rows = {k: score_group(v) for k, v in groups.items()}
    s = summary(rows)
    if len(rows) != 21 or s['views'] != 8814 or any(abs(s['means'][k]['auc']['v196_fused']-audit['summary'][k]['model_auc']) > .0002
                                                for k in ('trajectory', 'relative')):
        raise ValueError('Previous fused-score audit did not reproduce on this roster')
    write(OUT/'readout.json', dict(key=key, freeze=sha(freeze), summary=s,
        by_site_seed={f'{site}/{seed}': v for (site, seed), v in rows.items()}, original_2pct=True,
        new_allocation_executed=False, external_outcomes_read=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False), immutable=True)
    print(s, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze', 'score'))
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, source, original, parent, prior, audit = register()
        if args.phase == 'freeze':
            freeze(key, source, original, prior)
        else:
            score(key, source, original, parent, prior, audit)
