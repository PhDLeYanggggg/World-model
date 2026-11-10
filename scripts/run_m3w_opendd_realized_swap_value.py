"""Train direct action values and score allocations with unchanged nominations."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use native arm64 .venv-pytorch')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_recipient_query_mask as prior
from src.world_model import m3w_recipient_realized_value as model

old, base, np, torch = prior.old, prior.base, prior.np, prior.torch
read, write, sha = prior.read, prior.write, prior.sha
OUT = prior.OUT.parent/'opendd_realized_swap_value_v1'
LOCAL = prior.LOCAL.parent/OUT.name
ARMS = ('original', 'opportunity', 'trajectory', 'joint')
CONTRASTS = (('trajectory', 'opportunity'), ('joint', 'opportunity'), ('joint', 'trajectory'),
             ('opportunity', 'original'), ('trajectory', 'original'), ('joint', 'original'))
STEPS, SAVE, BATCH = 300, 50, 32


def beat(phase, **info):
    write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, **info))
    print(phase, info, flush=True)


def register():
    parent_key, source, original, _, _, _, _ = prior.register()
    f, r = prior.OUT/'freeze.json', prior.OUT/'readout.json'
    masks, results = read(f), read(r)
    if (masks['key'] != parent_key or results['freeze'] != sha(f) or not results['original_2pct']
            or results['external_outcomes_read'] or results['independent_confirmation']):
        raise ValueError('Require unchanged scored query masks')
    dest = OUT/'registration.json'
    refs = (f, r, Path(__file__), Path(model.__file__), OUT/'protocol.md')
    write(dest, dict(files={str(p.relative_to(base.prior.ROOT)): sha(p) for p in refs},
        seeds=base.SEEDS, sites=base.SITES, steps=STEPS, workers=0, threads=4, batch_per_donor=BATCH,
        checkpoint_every=SAVE, pair_weight=model.RELATIVE_WEIGHT, original_2pct=True,
        external_outcomes_read=False), immutable=True)
    return sha(dest), source, original, masks, results


def menus(packet, f, cv, anchor, bounds):
    for (lo, hi), (a, b, new, x) in zip(packet['bounds'], old.menus(packet, f, cv, anchor, bounds), strict=True):
        yield slice(lo, hi), a-lo, b-lo, new, x


def donors(held, seed, original, features):
    data, proof = {}, {}
    for name, c in original['cells'].items():
        if c['seed'] != seed or held not in c['packets']:
            continue
        old.check_cell(c)
        site, = set(c['packets'])-{held}
        if site in data:
            raise ValueError('Duplicate donor direction')
        f, orig = features[site], c['packets'][site]
        rows = []
        with prior.prior.packet(orig) as packet, np.load(f['file'], allow_pickle=False) as d:
            y, valid = d['future'], d['valid']
            cv, anchor, bounds = old.geometry(packet, f)
            for q, a, b, new, x in menus(packet, f, cv, anchor, bounds):
                if len(x):
                    inp, _, info = model.geometry(cv[q], anchor[q], a, b, new, x)
                    rows.append((inp, model.targets(y[q], valid[q], a, b, info)))
        data[site] = tuple(torch.from_numpy(np.concatenate(z)) for z in zip(*rows, strict=True))
        proof[site] = dict(cell=name, packet=orig['sha256'], cohort=sha(f['file']), candidates=len(data[site][0]))
    if set(data) != set(base.SITES)-{held}:
        raise ValueError('The outer held location cannot supply fitting targets')
    return [data[s] for s in sorted(data)], proof


def fit(identity, data):
    held, seed = identity['held'], identity['seed']
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, base.SITES.index(held)])
    net = model.network(data[0][0].shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    p = LOCAL/f'{held}_{seed}.pt'
    start = 0
    if p.exists():
        z = torch.load(p, map_location='cpu', weights_only=True)
        if z['identity'] != identity or not 0 <= z['step'] <= STEPS:
            raise ValueError('Changed checkpoint')
        start = z['step']
        net.load_state_dict(z['weights'])
        opt.load_state_dict(z['optimizer'])
        rng.bit_generator.state = z['rng']
    for step in range(start+1, STEPS+1):
        parts = [(x[i], y[i]) for x, y in data for i in (rng.integers(len(y), size=BATCH),)]
        x, y = (torch.cat(a) for a in zip(*parts, strict=True))
        loss = torch.nn.functional.smooth_l1_loss(net(x), y, beta=.2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            tmp = p.with_suffix('.pending')
            torch.save(dict(identity=identity, step=step, weights=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state), tmp)
            tmp.replace(p)
            beat('fit', held=held, seed=seed, step=step, loss=float(loss.detach()))
    return net.eval(), dict(file=p.name, sha256=sha(p), step=STEPS)


@torch.inference_mode()
def propose(packet, f, previous, net):
    cv, anchor, bounds = old.geometry(packet, f)
    action, cost = previous['actions'], previous['cost']
    n = len(packet['bounds'])
    if (action.shape != (n,) or cost.shape != (n,) or np.any((cost <= 0.) != (action < 0))):
        raise ValueError('Changed nominated action budget')
    data = np.zeros((n, 4), dtype=np.float32)
    scale = np.zeros((n, 2))
    for i, (q, a, b, new, x) in enumerate(menus(packet, f, cv, anchor, bounds)):
        j = int(action[i])
        if j < 0:
            if len(x):
                raise ValueError('Missing legal nomination')
            continue
        if j >= len(x):
            raise ValueError('Invalid nomination')
        s = slice(j, j+1)
        if not np.isclose(cost[i], np.linalg.norm(anchor[q][a[j]]-cv[q][a[j]], axis=-1).max(), atol=1e-10, rtol=0.):
            raise ValueError('Frozen movement cost changed')
        x, scale[i], _ = model.geometry(cv[q], anchor[q], a[s], b[s], new[s], x[s])
        data[i] = net(torch.from_numpy(x)).numpy()[0]
    pred = model.utility(data, scale)
    masks, status = {}, {}
    for a, u in zip(ARMS[2:], pred, strict=True):
        masks[a], status[a] = prior.model.allocate(u, cost, previous['archived_mask'])
    return dict(actions=action, cost=cost, budget_anchor=previous['archived_mask'],
                opportunity=previous['query_mask'], prediction=data, scale=scale, **masks), status


def freeze(key, source, original, previous):
    features, cells = old.prior.load_features(source), {}
    for held in base.SITES:
        for seed in base.SEEDS:
            tag = f'{held}_{seed}'
            parent = previous['cells'][tag]
            if held in parent['fit']:
                raise ValueError('Parent selector used held labels')
            f = LOCAL/f'{tag}.json'
            if not f.exists():
                train, proof = donors(held, seed, original, features)
                net, ckpt = fit(dict(key=key, held=held, seed=seed, donors=proof), train)
                packets = {}
                for name, v in original['cells'].items():
                    if v['seed'] != seed or held not in v['packets']:
                        continue
                    r = parent['packets'][name]
                    src, orig = prior.LOCAL/r['file'], v['packets'][held]
                    if sha(src) != r['sha256'] or r['original'] != orig['sha256']:
                        raise ValueError('Changed parent nominations')
                    with prior.prior.packet(orig) as p, np.load(src, allow_pickle=False) as n:
                        arrays, status = propose(p, features[held], n, net)
                    dest = LOCAL/f'{tag}_{name}.npz'
                    packets[name] = dict(file=dest.name, sha256=base.parent.indices.save_npz(dest, arrays),
                                         original=orig['sha256'], nomination=r['sha256'], status=status)
                write(f, dict(key=key, fit=proof, checkpoint=ckpt, packets=packets), immutable=True)
            v = read(f)
            if (v['key'] != key or held in v['fit'] or len(v['packets']) != 6 or
                    any(sha(LOCAL/r['file']) != r['sha256'] for r in (v['checkpoint'], *v['packets'].values()))):
                raise ValueError('Changed frozen actions')
            cells[tag] = v
            beat('freeze', completed=len(cells), total=21)
    write(OUT/'freeze.json', dict(key=key, cells=cells, decisions_frozen=True, external_outcomes_read=False), immutable=True)


def relative_contrast(pred, ref, y, valid):
    i, j = np.triu_indices(len(ref), 1)
    if not len(i):
        return np.zeros(3)
    known, lower = old.pair.pair_reduction(pred[None], ref, y, valid)[0]
    seen = known*float((valid[i] & valid[j]).mean())
    return np.array([seen, lower, 2.*seen-lower])


def evaluate(packet, f, d, cut, z):
    cv, anchor, _ = old.geometry(packet, f)
    y, valid, w = (d[k] for k in ('future', 'valid', 'weight'))
    n, cost = len(packet['bounds']), z['cost']
    keep = z['budget_anchor']
    budget, quota = float(cost[keep].sum()), int(keep.sum())
    masks = {a: z[a] for a in ARMS[1:]}
    spend = {a: float(cost[m].sum()) for a, m in masks.items()}
    if any(m.shape != (n,) or m.dtype != np.bool_ or int(m.sum()) != quota
           or not (.95-1e-7)*budget <= spend[a] <= (1.+1e-7)*budget for a, m in masks.items()):
        raise ValueError('Changed count or movement range')
    pred = {a: anchor.copy() for a in ARMS}
    rel = {f'{a}_minus_{b}': np.zeros(3) for a, b in CONTRASTS}
    counts = Counter(views=n, swaps=quota, **{f'{a}_strict': 0 for a in masks})
    for i, ((a, b, new, _), t) in enumerate(prior.prior.labels(packet, f, d, cut)):
        j = int(z['actions'][i])
        if j < 0 or not any(m[i] for m in masks.values()):
            continue
        strict = bool(t[j, 0] > 0. and t[j, 1] == 1.)
        for k, m in masks.items():
            if m[i]:
                pred[k][a[j]], pred[k][b[j]] = cv[a[j]], new[j]
                counts[f'{k}_strict'] += strict
        q = slice(*packet['bounds'][i])
        for a, b in CONTRASTS:
            if not np.array_equal(pred[a][q], pred[b][q]):
                rel[f'{a}_minus_{b}'] += relative_contrast(pred[a][q], pred[b][q], y[q], valid[q])
    use = np.isin(d['recording'], packet['records'])
    ys, mask, cv_used, rec = y[use], valid[use], cv[use], d['recording'][use]
    mass = float(w[use] @ np.where(mask, np.linalg.norm(cv_used-ys, axis=-1), 0.).sum(1))
    arms, effects = {}, {}
    for a, p in pred.items():
        row = old.pair.evaluate(p, packet, d, cut)
        pol = base.policy(p[use], cv_used, cv_used, ys, mask, rec, easy_cut=cut[0], hard_cut=cut[1])
        easy = [r['known_easy_net_degradation'] for r in pol['recordings'].values()
                if r['known_easy_net_degradation'] is not None]
        arms[a] = dict(**{k: int(row[k].sum()) for k in ('risk', 'dual', 'missing')},
            ade=pol['known_ADE_pixel'], fde=pol['available_endpoint_FDE_diagnostic_pixel'],
            p95=pol['p95_known_ADE_pixel'], worst_easy=100.*max(easy) if easy else None)
    for a, b in CONTRASTS:
        k = f'{a}_minus_{b}'
        known, lower = base.parent.paired(pred[b][use], pred[a][use], ys, mask)
        g, lo = (float(100.*(w[use] @ x.sum(1))/mass) for x in (known, lower))
        effects[k] = dict(known_pp=g, lower_pp=lo, upper_pp=2*g-lo,
            **dict(zip(('relative_observed', 'relative_lower', 'relative_upper'), (rel[k]/n).tolist(), strict=True)),
            dual_pp=float(100.*(arms[a]['dual']-arms[b]['dual'])/n))
    return dict(counts=dict(counts), arms=arms, effects=effects, spend=spend, budget=budget)


def summarize(rows):
    groups = {s: [r for r in rows if r['identity']['held'] == s] for s in base.SITES}
    draws = np.random.default_rng(199).integers(7, size=(3000, 7))

    def ci(key, metric):
        x = np.array([np.mean([np.mean([r['effects'][key][metric] for r in g if r['identity']['seed'] == s])
                               for s in base.SEEDS]) for g in groups.values()])
        return dict(point=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
                    by_site=dict(zip(groups, x.tolist(), strict=True)))

    arms = {}
    for a in ARMS:
        r = [v['arms'][a] for v in rows]
        arms[a] = {k: sum(v[k] for v in r) for k in ('risk', 'dual', 'missing')}
        for k in ('ade', 'fde', 'p95'):
            arms[a][f'direction_mean_{k}'] = float(np.mean([v[k] for v in r if v[k] is not None]))
        arms[a]['worst_easy'] = max(v['worst_easy'] for v in r if v['worst_easy'] is not None)
    return dict(counts=dict(sum((Counter(r['counts']) for r in rows), Counter())), arms=arms,
                spend={k: float(sum(r['spend'][k] for r in rows)) for k in ARMS[1:]},
                budget=float(sum(r['budget'] for r in rows)),
                paired={a: {m: ci(a, m) for m in v} for a, v in rows[0]['effects'].items()})


def score(key, source, original, parent_results):
    p = OUT/'freeze.json'
    old.require_published(p)
    frozen = read(p)
    if frozen['key'] != key or not frozen['decisions_frozen'] or frozen['external_outcomes_read']:
        raise ValueError('Require a published matched action freeze')
    features, data = old.prior.load_features(source), base.parent.load(source)
    rows, solver = [], Counter()
    for name, c in original['cells'].items():
        old.check_cell(c)
        for held, orig in c['packets'].items():
            seed = c['seed']
            cell = frozen['cells'][f'{held}_{seed}']
            ref = cell['packets'][name]
            action = LOCAL/ref['file']
            if held in cell['fit'] or ref['original'] != orig['sha256'] or sha(action) != ref['sha256']:
                raise ValueError('Changed held scores')
            solver.update(f'{k}_{v}' for k, v in ref['status'].items())
            f = LOCAL/f'{name}_{held}_score.json'
            ident = dict(key=key, held=held, seed=seed, packet=orig['sha256'], action=ref['sha256'])
            if not f.exists():
                with prior.prior.packet(orig) as p, np.load(action, allow_pickle=False) as z:
                    v = evaluate(p, features[held], data[held], c['cut'], z)
                write(f, dict(identity=ident, **v), immutable=True)
            row = read(f)
            if row['identity'] != ident:
                raise ValueError('Changed resumed score')
            rows.append(row)
        beat('score', completed=len(rows), total=126)
    s = summarize(rows)
    b, c = parent_results['summary'], s['counts']
    if (len(rows) != 126 or c['swaps'] != b['counts']['swaps'] or c['views'] != b['counts']['views']
            or c['opportunity_strict'] != b['counts']['query_mask_qualifying']
            or any(s['arms']['opportunity'][k] != b['arms']['query_mask'][k] for k in ('risk', 'dual', 'missing'))):
        raise ValueError('The existing opportunity rule did not replay')
    write(OUT/'readout.json', dict(key=key, freeze=sha(OUT/'freeze.json'), summary=s, solver=dict(solver),
        by_direction=rows, external_outcomes_read=False, independent_confirmation=False,
        deployment_changed=False, original_2pct=True), immutable=True)
    print(c, dict(solver), flush=True)


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
        key, source, original, parent, result = register()
        if args.phase == 'freeze':
            freeze(key, source, original, parent)
        else:
            score(key, source, original, result)
