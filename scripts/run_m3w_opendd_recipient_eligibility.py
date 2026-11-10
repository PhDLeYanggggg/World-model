"""Fit and test swap eligibility at the archived, identical query mask."""

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
from scripts import audit_m3w_recipient_transport_ceiling as ceiling
from scripts import run_m3w_opendd_recipient_transport as old
from src.world_model import m3w_recipient_eligibility as model

np, torch, base = old.np, old.torch, old.base
read, write, sha = old.read, old.write, old.sha
OUT = old.OUT.parent/'opendd_recipient_eligibility_v1'
LOCAL = old.LOCAL.parent/OUT.name
ARMS = ('original', 'archived', 'gain', 'eligibility', 'guarded')
CONTRASTS = ((a, 'original') for a in ARMS[1:])
CONTRASTS = (*CONTRASTS, ('eligibility', 'gain'), ('eligibility', 'archived'))
STEPS, SAVE, BATCH = 300, 50, 32


def register():
    parent, source, original, _ = old.register()
    p, r, oracle = (old.OUT/f'{k}.json' for k in ('freeze', 'readout', 'ceiling'))
    frozen, scored = read(p), read(r)
    if (frozen['registration'] != parent or scored['freeze'] != sha(p)
            or not scored['original_2pct'] or scored['external_outcomes_read']):
        raise ValueError('Require the scored, bound transport comparison')
    files = (p, r, oracle, Path(__file__), Path(model.__file__), Path(ceiling.__file__), OUT/'protocol.md')
    q = OUT/'registration.json'
    write(q, dict(files={str(f.relative_to(base.prior.ROOT)): sha(f) for f in files},
                  seeds=base.SEEDS, sites=base.SITES, threads=4, workers=0,
                  steps=STEPS, checkpoint_every=SAVE, batch_per_site=BATCH,
                  inference_probability_threshold=model.THRESHOLD, external_outcomes_read=False), immutable=True)
    return sha(q), source, original, frozen, scored, read(oracle)


def packet(c):
    return old.ref.prior.packet(old.prior.study.LOCAL, c)


def prior_actions(frozen, held, seed, name, orig):
    cell = frozen['cells'][f'{held}_{seed}']
    r = cell['packets'][name]
    path = old.LOCAL/r['file']
    if r['original'] != orig['sha256'] or held in cell['fit_sites'] or r['sha256'] != sha(path):
        raise ValueError('Changed archived mask or held-role boundary')
    with np.load(path, allow_pickle=False) as z:
        return z['mean'], r['sha256']


def labels(p, f, d, cut):
    """Materialize donor or post-freeze scoring labels, never inference inputs."""
    cv, anchor, bounds = old.geometry(p, f)
    ols = p['ols4']
    y, v, w = (d[k] for k in ('future', 'valid', 'weight'))
    names, rec = np.unique(d['recording'], return_inverse=True)
    idx = np.searchsorted(names, p['records'])
    np.testing.assert_array_equal(names[idx], p['records'])
    moments = ceiling.moments
    cap = .02*np.bincount(rec, weights=moments(cv, cv, y, v, cut[0])['ref_easy'])
    paired = base.parent.paired
    ols_base = paired(ols, cv, y, v)[1].sum(1)
    ols_offset = np.bincount(rec, weights=w*ols_base)
    for i, (a, b, new, x) in enumerate(old.menus(p, f, cv, anchor, bounds)):
        lo, hi = p['bounds'][i]
        if not np.all(rec[lo:hi] == idx[i]):
            raise ValueError('Query spans recordings')
        r = p['additive_rows'][i]
        keep = np.where(a == r[0], r[1], r[0])
        rows = np.column_stack((keep, b))
        use = rows.flatten()
        forecasts = np.stack((anchor[keep], new), axis=1).reshape(-1, *cv.shape[1:])
        m = {k: z.reshape(rows.shape) for k, z in moments(cv[use], forecasts, y[use], v[use], cut[0]).items()}
        gain_cv = (w[use]*paired(cv[use], forecasts, y[use], v[use])[1].sum(1)).reshape(rows.shape).sum(1)
        gain_ols = (w[use]*(paired(ols[use], forecasts, y[use], v[use])[1].sum(1)-ols_base[use]))
        gain_ols = gain_ols.reshape(rows.shape).sum(1)+ols_offset[idx[i]]
        eligible = ceiling.safe(m, cap[idx[i]]) & (gain_cv > 0.) & (gain_ols > 0.)
        gain = old.model.targets(cv, anchor, y, v, a, b, new)
        yield (a, b, new, x), np.column_stack((gain, eligible)).astype(np.float32)


def donors(held, seed, original, features):
    out, proof = {}, {}
    for name, cell in original['cells'].items():
        if cell['seed'] != seed or held not in cell['packets']:
            continue
        old.check_cell(cell)
        s, = set(cell['packets'])-{held}
        f, src = features[s], cell['packets'][s]
        if s in out:
            raise ValueError('Duplicate donor')
        with np.load(f['file'], allow_pickle=False) as d, packet(src) as p:
            rows = [(x, y) for (*_, x), y in labels(p, f, d, cell['cut']) if len(y)]
        x, y = (np.concatenate(z) for z in zip(*rows, strict=True))
        out[s] = tuple(map(torch.from_numpy, (x, y)))
        proof[s] = dict(pair=name, packet=src['sha256'], cohort=sha(f['file']), rows=len(y),
                        eligible_rows=int(y[:, 1].sum()))
    if set(out) != set(base.SITES)-{held}:
        raise ValueError('The fitting cohort must exclude the held source')
    return [out[s] for s in sorted(out)], proof


def fit(identity, d):
    held, seed = identity['held'], identity['seed']
    rng = np.random.default_rng([seed, base.SITES.index(held)])
    torch.manual_seed(seed)
    net = model.network(d[0][0].shape[1])
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
    p = LOCAL/f'{held}_{seed}.pt'
    start = 0
    if p.exists():
        state = torch.load(p, map_location='cpu', weights_only=True)
        if state['identity'] != identity or not 0 <= state['step'] <= STEPS:
            raise ValueError('Checkpoint lineage or update changed')
        start = state['step']
        rng.bit_generator.state = state['rng']
        net.load_state_dict(state['weights'])
        optimizer.load_state_dict(state['optimizer'])
    for step in range(start+1, STEPS+1):
        batch = [(x[i], y[i]) for x, y in d for i in (rng.integers(len(x), size=BATCH),)]
        x, y = (torch.cat(z) for z in zip(*batch, strict=True))
        loss, gain, risk = model.loss(net(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step % SAVE == 0:
            tmp = p.with_suffix('.pending')
            torch.save(dict(identity=identity, step=step, rng=rng.bit_generator.state,
                            weights=net.state_dict(), optimizer=optimizer.state_dict()), tmp)
            tmp.replace(p)
            beat('fit', held=held, seed=seed, step=step, gain_loss=float(gain.detach()),
                 eligibility_loss=float(risk.detach()))
    return net.eval(), dict(file=p.name, sha256=sha(p))


def beat(phase, **info):
    write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, **info))
    print(phase, info, flush=True)


@torch.inference_mode()
def propose(p, f, net, archived):
    cv, anchor, bounds = old.geometry(p, f)
    if archived.shape != (len(p['bounds']),):
        raise ValueError('Incorrect archived mask shape')
    selected = np.full((3, len(archived)), -1, dtype=np.int32)
    probabilities = np.full((3, len(archived)), -1., dtype=np.float32)
    for i, (a, b, new, x) in enumerate(old.menus(p, f, cv, anchor, bounds)):
        if archived[i] < 0:
            continue
        if archived[i] >= len(x):
            raise ValueError('Archived action outside the legal menu')
        z = net(torch.from_numpy(x))
        distance = (np.linalg.norm(anchor[a]-cv[a], axis=-1).sum(1)
                    + np.linalg.norm(new-cv[b], axis=-1).sum(1))
        prob = z[:, 1].sigmoid().numpy()
        selected[:, i] = model.rank(z[:, 0].numpy(), prob, distance)
        for j, action in enumerate(selected[:, i]):
            if action >= 0:
                probabilities[j, i] = prob[action]
    return dict(archived=archived, probability=probabilities, **dict(zip(ARMS[2:], selected, strict=True)))


def freeze(key, source, original, prior):
    features, cells = old.prior.load_features(source), {}
    for held in base.SITES:
        for seed in base.SEEDS:
            tag = f'{held}_{seed}'
            receipt = LOCAL/f'{tag}.json'
            if not receipt.exists():
                data, lineage = donors(held, seed, original, features)
                net, ckpt = fit(dict(key=key, held=held, seed=seed, donors=lineage), data)
                decisions = {}
                for k, c in original['cells'].items():
                    if c['seed'] != seed or held not in c['packets']:
                        continue
                    old.check_cell(c)
                    p = c['packets'][held]
                    action, sh = prior_actions(prior, held, seed, k, p)
                    with packet(p) as z:
                        a = propose(z, features[held], net, action)
                    f = LOCAL/f'{tag}_{k}.npz'
                    decisions[k] = dict(file=f.name, sha256=base.parent.indices.save_npz(f, a),
                                        original=p['sha256'], prior_actions=sh)
                write(receipt, dict(key=key, checkpoint=ckpt, fit=lineage, decisions=decisions), immutable=True)
            c = read(receipt)
            if (c['key'] != key or set(c['fit']) != set(base.SITES)-{held} or len(c['decisions']) != 6
                    or any(sha(LOCAL/f['file']) != f['sha256'] for f in (c['checkpoint'], *c['decisions'].values()))):
                raise ValueError('Changed fit or frozen decisions')
            cells[tag] = c
            beat('freeze', completed=len(cells), total=21)
    write(OUT/'freeze.json', dict(key=key, cells=cells, predictions_frozen=True, external_outcomes_read=False), immutable=True)


def evaluate(p, f, d, cut, z):
    cv, anchor, _ = old.geometry(p, f)
    y, valid, w = (d[k] for k in ('future', 'valid', 'weight'))
    n = len(p['bounds'])
    edited = z['archived'] >= 0
    if any(z[k].shape != (n,) for k in ARMS[1:]) or not (
            np.array_equal(z['gain'] >= 0, edited) and np.array_equal(z['eligibility'] >= 0, edited)
            and np.all((z['guarded'] < 0) | (z['guarded'] == z['eligibility']))):
        raise ValueError('Matched arms changed the swap count or query mask')
    paths = {k: anchor.copy() for k in ARMS}
    counts = Counter(views=n)
    counts.update({f'{k}_{m}': 0 for k in ARMS[1:] for m in ('swaps', 'strict', 'eligible')})
    relative = {f'{a}_minus_{b}': 0. for a, b in CONTRASTS}
    brier = {k: [] for k in ARMS[2:]}
    for i, ((a, b, new, _), outcomes) in enumerate(labels(p, f, d, cut)):
        strict = (outcomes[:, 0] > 0.) & (outcomes[:, 1] == 1.)
        counts['opportunity_all'] += bool(strict.any())
        if not edited[i]:
            continue
        counts['opportunity_on_mask'] += bool(strict.any())
        counts['risk_ranking_differs'] += int(z['gain'][i] != z['eligibility'][i])
        for k in ARMS[1:]:
            j = z[k][i]
            if j < 0:
                continue
            paths[k][a[j]], paths[k][b[j]] = cv[a[j]], new[j]
            counts[f'{k}_swaps'] += 1
            counts[f'{k}_strict'] += int(strict[j])
            counts[f'{k}_eligible'] += int(outcomes[j, 1])
            if k in brier:
                brier[k].append(float((z['probability'][ARMS[2:].index(k), i]-outcomes[j, 1])**2))
        lo, hi = p['bounds'][i]
        q = slice(lo, hi)
        for a, b in CONTRASTS:
            if not np.array_equal(paths[a][q], paths[b][q]):
                relative[f'{a}_minus_{b}'] += old.pair.pair_reduction(paths[a][None, q], paths[b][q], y[q], valid[q])[0, 1]

    arms = {}
    use = np.isin(d['recording'], p['records'])
    rec = d['recording'][use]
    mask, future, floor = valid[use], y[use], cv[use]
    known, end = mask.any(1), mask[:, -1]
    mass = float(w[use] @ np.where(mask, np.linalg.norm(floor-future, axis=-1), 0.).sum(1))
    for k, pred in paths.items():
        val = old.pair.evaluate(pred, p, d, cut)
        arms[k] = {a: int(val[a].sum()) for a in ('risk', 'dual', 'missing')}
        pol = base.policy(pred[use], floor, floor, future, mask, rec, easy_cut=cut[0], hard_cut=cut[1])
        err = np.linalg.norm(pred[use]-future, axis=-1)
        easy = [t['known_easy_net_degradation'] for t in pol['recordings'].values()
                if t['known_easy_net_degradation'] is not None]
        arms[k].update(known_ade=pol['known_ADE_pixel'], endpoint_fde=float(err[end, -1].mean()) if end.any() else None,
                       p95=pol['p95_known_ADE_pixel'], worst_record_easy_pct=100.*max(easy) if easy else None,
                       easy_gain_pct=pol['subsets']['easy']['gain_percent'], hard_gain_pct=pol['subsets']['hard']['gain_percent'],
                       known_rows=int(known.sum()), endpoint_rows=int(end.sum()),
                       brier_sum=float(sum(brier.get(k, []))))
    effects = {}
    for a, b in CONTRASTS:
        k = f'{a}_minus_{b}'
        g, lower = base.parent.paired(paths[b][use], paths[a][use], future, mask)
        effects[k] = dict(known_pp=float(100.*(w[use] @ g.sum(1))/mass),
                          lower_pp=float(100.*(w[use] @ lower.sum(1))/mass),
                          relative_lower=float(relative[k]/n),
                          dual_pp=float(100.*(arms[a]['dual']-arms[b]['dual'])/n))
    return dict(arms=arms, effects=effects, counts=dict(counts))


def summary(rows):
    groups = {s: [r for r in rows if r['identity']['held'] == s] for s in base.SITES}
    draw = np.random.default_rng(196).integers(len(groups), size=(3000, len(groups)))

    def ci(c, m):
        x = np.asarray([np.mean([np.mean([r['effects'][c][m] for r in v if r['identity']['seed'] == seed])
                                 for seed in base.SEEDS]) for v in groups.values()])
        return dict(point=float(x.mean()), ci95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    by_site=dict(zip(groups, x.tolist(), strict=True)))

    counts = dict(sum((Counter(r['counts']) for r in rows), Counter()))
    arms = {}
    for a in ARMS:
        val = [r['arms'][a] for r in rows]
        x = {k: sum(r[k] for r in val) for k in ('risk', 'dual', 'missing')}
        for k in ('known_ade', 'endpoint_fde', 'p95'):
            x[f'direction_mean_{k}'] = float(np.mean([r[k] for r in val if r[k] is not None]))
        x['worst_record_easy_pct'] = max(r['worst_record_easy_pct'] for r in val if r['worst_record_easy_pct'] is not None)
        if a in ARMS[2:] and counts.get(f'{a}_swaps'):
            x['selected_eligibility_brier'] = sum(r['brier_sum'] for r in val)/counts[f'{a}_swaps']
        arms[a] = x
    return dict(counts=counts, arms=arms, paired={c: {m: ci(c, m) for m in z} for c, z in rows[0]['effects'].items()})


def score(key, source, original, prior, old_score, oracle):
    receipt = OUT/'freeze.json'
    old.require_published(receipt)
    frozen = read(receipt)
    if frozen['key'] != key or not frozen['predictions_frozen'] or frozen['external_outcomes_read']:
        raise ValueError('Require registered pre-score freeze')
    features, data = old.prior.load_features(source), base.parent.load(source)
    rows = []
    for name, c in original['cells'].items():
        old.check_cell(c)
        for held, orig in c['packets'].items():
            seed = c['seed']
            cell = frozen['cells'][f'{held}_{seed}']
            r = cell['decisions'][name]
            p, s = prior_actions(prior, held, seed, name, orig)
            f = LOCAL/r['file']
            if r['original'] != orig['sha256'] or r['sha256'] != sha(f) or s != r['prior_actions'] or held in cell['fit']:
                raise ValueError('Changed prediction binding')
            local = LOCAL/f'{name}_{held}_score.json'
            ident = dict(key=key, held=held, seed=seed, packet=orig['sha256'], action=r['sha256'])
            if not local.exists():
                with packet(orig) as z, np.load(f, allow_pickle=False) as actions:
                    np.testing.assert_array_equal(actions['archived'], p)
                    v = evaluate(z, features[held], data[held], c['cut'], actions)
                write(local, dict(identity=ident, **v), immutable=True)
            row = read(local)
            if row['identity'] != ident:
                raise ValueError('Changed resumed result')
            rows.append(row)
        beat('score', completed=len(rows), total=126)
    s = summary(rows)
    counts, arms = s['counts'], s['arms']
    prior_count = old_score['summary']['counts']
    if (len(rows) != 126 or counts['views'] != prior_count['views'] or counts['archived_swaps'] != prior_count['mean_swaps']
            or counts['opportunity_all'] != oracle['totals']['strict_opportunity']
            or counts['archived_strict'] != oracle['totals']['mean_strict']
            or any(arms['archived'][k] != old_score['summary']['arms']['mean'][k] for k in ('risk', 'dual', 'missing'))):
        raise ValueError('Exact risk, matched mask or prior hindsight results did not reproduce')
    write(OUT/'readout.json', dict(key=key, freeze=sha(receipt), summary=s, by_direction=rows,
                                  external_outcomes_read=False, original_2pct=True,
                                  independent_confirmation=False, deployment_changed=False), immutable=True)
    print(counts, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'score'))
    arg = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        k, source, original, prior, r, oracle = register()
        if arg.phase == 'freeze':
            freeze(k, source, original, prior)
        else:
            score(k, source, original, prior, r, oracle)
