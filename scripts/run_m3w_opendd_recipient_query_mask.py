"""Fit a query opportunity head and compare budget-matched masks."""

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
from scripts import run_m3w_opendd_recipient_eligibility as prior
from src.world_model import m3w_recipient_query_mask as model

old, base, np, torch = prior.old, prior.base, prior.np, prior.torch
read, write, sha = prior.read, prior.write, prior.sha
OUT = prior.OUT.parent/'opendd_recipient_query_mask_v1'
LOCAL = prior.LOCAL.parent/OUT.name
MASKS = ('archived_mask', 'gain_mask', 'query_mask')
ARMS = ('original', *MASKS)
CONTRASTS = tuple((a, 'original') for a in MASKS)+(
    ('gain_mask', 'archived_mask'), ('query_mask', 'archived_mask'), ('query_mask', 'gain_mask'))
STEPS, SAVE, BATCH = 300, 50, 32


def beat(phase, **info):
    write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, **info))
    print(phase, info, flush=True)


def register():
    key, source, original, transport, _, oracle = prior.register()
    f, r = prior.OUT/'freeze.json', prior.OUT/'readout.json'
    freeze, readout = read(f), read(r)
    if (freeze['key'] != key or readout['freeze'] != sha(f) or readout['external_outcomes_read']
            or readout['independent_confirmation'] or not readout['original_2pct']):
        raise ValueError('Require the scored previous development round')
    q = OUT/'registration.json'
    files = (f, r, Path(__file__), Path(model.__file__), OUT/'protocol.md')
    write(q, dict(files={str(p.relative_to(base.prior.ROOT)): sha(p) for p in files},
                  seeds=base.SEEDS, sites=base.SITES, steps=STEPS, batch_per_donor=BATCH,
                  threads=4, workers=0, checkpoint_every=SAVE, minimum_spend=model.LOWER_SHARE,
                  original_2pct=True, external_outcomes_read=False), immutable=True)
    return sha(q), source, original, transport, freeze, readout, oracle


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
        with prior.packet(orig) as p, np.load(f['file'], allow_pickle=False) as d:
            rows = [(model.describe(x), bool(((y[:, 0] > 0.) & (y[:, 1] == 1.)).any()))
                    for (*_, x), y in prior.labels(p, f, d, c['cut']) if len(y)]
        x, y = (np.stack(z) for z in zip(*rows, strict=True))
        data[site] = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
        proof[site] = dict(cell=name, packet=orig['sha256'], cohort=sha(f['file']),
                           queries=len(y), positives=int(y.sum()))
    if set(data) != set(base.SITES)-{held}:
        raise ValueError('Held source cannot contribute fitting labels')
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
            raise ValueError('Changed checkpoint identity')
        start = z['step']
        net.load_state_dict(z['weights'])
        opt.load_state_dict(z['optimizer'])
        rng.bit_generator.state = z['rng']
    for step in range(start+1, STEPS+1):
        parts = [(x[i], y[i]) for x, y in data for i in (rng.integers(len(y), size=BATCH),)]
        x, y = (torch.cat(a) for a in zip(*parts, strict=True))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), y)
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
def propose(p, f, local_head, query_head, archived):
    cv, anchor, bounds = old.geometry(p, f)
    n = len(archived)
    if n != len(p['bounds']):
        raise ValueError('Changed archived mask')
    action, cost, gain = np.full(n, -1, dtype=np.int32), np.zeros(n), np.zeros(n)
    q = []
    for i, (a, b, new, x) in enumerate(old.menus(p, f, cv, anchor, bounds)):
        q.append(model.describe(x))
        if not len(x):
            continue
        delta = np.linalg.norm(anchor[a]-cv[a], axis=-1)
        distance = delta.sum(1)+np.linalg.norm(new-cv[b], axis=-1).sum(1)
        raw = local_head(torch.from_numpy(x))[:, 0].numpy()*distance
        j = int(raw.argmax())
        action[i], cost[i], gain[i] = j, delta[j].max(), raw[j]
    score = query_head(torch.from_numpy(np.stack(q))).sigmoid().numpy()
    masks = [archived >= 0]
    status = {}
    for k, value in zip(MASKS[1:], (gain, score), strict=True):
        chosen, status[k] = model.allocate(value, cost, masks[0])
        masks.append(chosen)
    return dict(actions=action, cost=cost, query_score=score, raw_gain=gain,
                **dict(zip(MASKS, masks, strict=True))), status


def freeze(key, source, original, transport, parent):
    data, cells = old.prior.load_features(source), {}
    for held in base.SITES:
        for seed in base.SEEDS:
            tag = f'{held}_{seed}'
            ref = parent['cells'][tag]
            cp = prior.LOCAL/ref['checkpoint']['file']
            if held in ref['fit'] or ref['checkpoint']['sha256'] != sha(cp):
                raise ValueError('Within-query chooser must exclude the held site')
            receipt = LOCAL/f'{tag}.json'
            if not receipt.exists():
                donors_data, proof = donors(held, seed, original, data)
                net, ckpt = fit(dict(key=key, held=held, seed=seed, donors=proof), donors_data)
                f = data[held]
                width = f['x'].shape[1]*3+5+24
                local = prior.model.network(width).eval()
                local.load_state_dict(torch.load(cp, map_location='cpu', weights_only=True)['weights'])
                packets = {}
                for name, c in original['cells'].items():
                    if c['seed'] != seed or held not in c['packets']:
                        continue
                    orig = c['packets'][held]
                    a, digest = prior.prior_actions(transport, held, seed, name, orig)
                    earlier = ref['decisions'][name]
                    ep = prior.LOCAL/earlier['file']
                    if sha(ep) != earlier['sha256'] or earlier['original'] != orig['sha256']:
                        raise ValueError('Wrong previous frozen nominations')
                    with prior.packet(orig) as p, np.load(ep, allow_pickle=False) as b:
                        arrays, status = propose(p, f, local, net, a)
                        np.testing.assert_array_equal(np.where(a >= 0, arrays['actions'], -1), b['gain'])
                    dest = LOCAL/f'{tag}_{name}.npz'
                    packets[name] = dict(file=dest.name, sha256=base.parent.indices.save_npz(dest, arrays),
                                         original=orig['sha256'], archived=digest, status=status)
                write(receipt, dict(key=key, fixed_chooser=sha(cp), checkpoint=ckpt, fit=proof, packets=packets), immutable=True)
            v = read(receipt)
            if (v['key'] != key or v['fixed_chooser'] != sha(cp) or held in v['fit'] or len(v['packets']) != 6
                    or any(sha(LOCAL/r['file']) != r['sha256'] for r in (v['checkpoint'], *v['packets'].values()))):
                raise ValueError('Invalid frozen mask lineage')
            cells[tag] = v
            beat('freeze', completed=len(cells), total=21)
    write(OUT/'freeze.json', dict(key=key, cells=cells, decisions_frozen=True, external_outcomes_read=False), immutable=True)


def evaluate(p, f, d, cut, z):
    cv, anchor, _ = old.geometry(p, f)
    y, valid, w = (d[k] for k in ('future', 'valid', 'weight'))
    n, cost = len(p['bounds']), z['cost']
    masks = {k: z[k] for k in MASKS}
    budget, quota = float(cost[masks[MASKS[0]]].sum()), int(masks[MASKS[0]].sum())
    paths = {a: anchor.copy() for a in ARMS}
    cnt = Counter(views=n, swaps=quota)
    spend = {k: float(cost[m].sum()) for k, m in masks.items()}
    if any(m.shape != (n,) or m.dtype != np.bool_ or int(m.sum()) != quota
           or not (model.LOWER_SHARE-1e-7)*budget <= spend[k] <= (1.+1e-7)*budget
           for k, m in masks.items()):
        raise ValueError('Count or spend is not matched')
    rel = {f'{a}_minus_{b}': 0. for a, b in CONTRASTS}
    qualifying = np.zeros(n, dtype=bool)
    opportunities = np.zeros(n, dtype=bool)
    for i, ((a, b, new, _), targets) in enumerate(prior.labels(p, f, d, cut)):
        strict = (targets[:, 0] > 0.) & (targets[:, 1] == 1.)
        opportunities[i] = strict.any()
        j = z['actions'][i]
        if j < 0:
            if len(strict) or any(m[i] for m in masks.values()):
                raise ValueError('Invalid empty menu decision')
            continue
        qualifying[i] = strict[j]
        for k, m in masks.items():
            if m[i]:
                paths[k][a[j]], paths[k][b[j]] = cv[a[j]], new[j]
        lo, hi = p['bounds'][i]
        q = slice(lo, hi)
        for a, b in CONTRASTS:
            if not np.array_equal(paths[a][q], paths[b][q]):
                rel[f'{a}_minus_{b}'] += old.pair.pair_reduction(paths[a][None, q], paths[b][q], y[q], valid[q])[0, 1]
    cnt.update(opportunities=int(opportunities.sum()), local_chooser_qualifies=int(qualifying.sum()))
    for k, m in masks.items():
        cnt[f'{k}_opportunities'] = int((m & opportunities).sum())
        cnt[f'{k}_qualifying'] = int((m & qualifying).sum())
        cnt[f'{k}_changed_queries'] = int((m & ~masks[MASKS[0]]).sum())
    hindsight, flag = model.allocate(qualifying.astype(float), cost, masks[MASKS[0]])
    cnt['verified_hindsight_qualifying'] = int((hindsight & qualifying).sum())
    cnt[f'hindsight_{flag}'] += 1
    use = np.isin(d['recording'], p['records'])
    ys, v, floor, rec = y[use], valid[use], cv[use], d['recording'][use]
    mass = float(w[use] @ np.where(v, np.linalg.norm(floor-ys, axis=-1), 0.).sum(1))
    arms = {}
    for a, pred in paths.items():
        val = old.pair.evaluate(pred, p, d, cut)
        pol = base.policy(pred[use], floor, floor, ys, v, rec, easy_cut=cut[0], hard_cut=cut[1])
        easy = [r['known_easy_net_degradation'] for r in pol['recordings'].values()
                if r['known_easy_net_degradation'] is not None]
        arms[a] = dict(**{k: int(val[k].sum()) for k in ('risk', 'dual', 'missing')},
                       ade=pol['known_ADE_pixel'], fde=pol['available_endpoint_FDE_diagnostic_pixel'],
                       p95=pol['p95_known_ADE_pixel'], worst_easy_pct=100.*max(easy) if easy else None)
    effects = {}
    for a, b in CONTRASTS:
        k = f'{a}_minus_{b}'
        known, lower = base.parent.paired(paths[b][use], paths[a][use], ys, v)
        effects[k] = dict(lower_pp=float(100.*(w[use] @ lower.sum(1))/mass),
                          known_pp=float(100.*(w[use] @ known.sum(1))/mass),
                          relative_lower=float(rel[k]/n), dual_pp=float(100.*(arms[a]['dual']-arms[b]['dual'])/n))
    return dict(counts=dict(cnt), spend=spend, effects=effects, arms=arms)


def summarize(rows):
    groups = {s: [r for r in rows if r['identity']['held'] == s] for s in base.SITES}
    draw = np.random.default_rng(198).integers(7, size=(3000, 7))

    def ci(a, b):
        x = np.array([np.mean([np.mean([r['effects'][a][b] for r in group if r['identity']['seed'] == seed])
                               for seed in base.SEEDS]) for group in groups.values()])
        return dict(point=float(x.mean()), ci95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    by_site=dict(zip(groups, x.tolist(), strict=True)))

    counts = {s: dict(sum((Counter(r['counts']) for r in g), Counter())) for s, g in groups.items()}
    arms = {}
    for a in ARMS:
        v = [r['arms'][a] for r in rows]
        arms[a] = {k: sum(r[k] for r in v) for k in ('risk', 'dual', 'missing')}
        for k in ('ade', 'fde', 'p95'):
            arms[a][f'direction_mean_{k}'] = float(np.mean([r[k] for r in v if r[k] is not None]))
        arms[a]['worst_easy_pct'] = max(r['worst_easy_pct'] for r in v if r['worst_easy_pct'] is not None)
    return dict(counts=dict(sum(map(Counter, counts.values()), Counter())), by_site_counts=counts, arms=arms,
                spend={a: float(sum(r['spend'][a] for r in rows)) for a in MASKS},
                paired={k: {m: ci(k, m) for m in v} for k, v in rows[0]['effects'].items()})


def score(key, source, original, parent_readout, oracle):
    receipt = OUT/'freeze.json'
    old.require_published(receipt)
    frozen = read(receipt)
    if frozen['key'] != key or not frozen['decisions_frozen'] or frozen['external_outcomes_read']:
        raise ValueError('Require a published mask freeze')
    features, data = old.prior.load_features(source), base.parent.load(source)
    rows, solver = [], Counter()
    for name, c in original['cells'].items():
        old.check_cell(c)
        for held, orig in c['packets'].items():
            seed = c['seed']
            cell = frozen['cells'][f'{held}_{seed}']
            r = cell['packets'][name]
            p = LOCAL/r['file']
            if held in cell['fit'] or r['original'] != orig['sha256'] or sha(p) != r['sha256']:
                raise ValueError('Wrong frozen mask or model lineage')
            solver.update(f'{k}_{v}' for k, v in r['status'].items())
            ident = dict(key=key, held=held, seed=seed, packet=orig['sha256'], decision=r['sha256'])
            f = LOCAL/f'{name}_{held}_score.json'
            if not f.exists():
                with prior.packet(orig) as z, np.load(p, allow_pickle=False) as a:
                    result = evaluate(z, features[held], data[held], c['cut'], a)
                write(f, dict(identity=ident, **result), immutable=True)
            out = read(f)
            if out['identity'] != ident:
                raise ValueError('Resumed score does not bind the original mask')
            rows.append(out)
        beat('score', completed=len(rows), total=126)
    s = summarize(rows)
    n, b = s['counts'], parent_readout['summary']
    if (len(rows) != 126 or n['views'] != b['counts']['views'] or n['swaps'] != b['counts']['gain_swaps']
            or n['archived_mask_qualifying'] != b['counts']['gain_strict']
            or n['archived_mask_opportunities'] != b['counts']['opportunity_on_mask']
            or n['opportunities'] != oracle['totals']['strict_opportunity']
            or any(s['arms']['archived_mask'][k] != b['arms']['gain'][k] for k in ('risk', 'dual', 'missing'))):
        raise ValueError('Archived mask, gains or exact ceiling did not reproduce')
    write(OUT/'readout.json', dict(key=key, freeze=sha(receipt), summary=s, solver=dict(solver), by_direction=rows,
        original_2pct=True, deployment_changed=False, independent_confirmation=False, external_outcomes_read=False), immutable=True)
    print(n, dict(solver), flush=True)


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
        key, source, original, transport, parent, prior_score, oracle = register()
        if args.phase == 'freeze':
            freeze(key, source, original, transport, parent)
        else:
            score(key, source, original, prior_score, oracle)
