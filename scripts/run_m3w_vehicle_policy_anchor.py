"""Test policy-selected donor actors under identical path families and held nominees."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import fit_m3w_vehicle_nested_additive as additive
from src.world_model import m3w_vehicle_policy_anchor as model

family, nested, action, ctx, base, np, torch = (
    additive.family, additive.nested, additive.action, additive.ctx, additive.base, additive.np, additive.torch)
prior, screen = family.prior, family.screen
sha, read, write, OUT, LOCAL = additive.sha, additive.read, additive.write, additive.OUT, additive.LOCAL
NEW = ('random_pointwise', 'policy_pointwise', 'random_budget', 'policy_budget')
ARMS = ('margin', *NEW)
PAIRS = (('policy_pointwise', 'random_pointwise'), ('policy_budget', 'random_budget'),
         ('policy_pointwise', 'margin'), ('policy_budget', 'margin'),
         ('random_pointwise', 'margin'), ('random_budget', 'margin'))


def register():
    key0, source, bank, parent, _, teachers = additive.register()
    af, pf = OUT/'additive_freeze.json', family.OUT/'freeze.json'
    teacher, held = read(af), read(pf)
    if (teacher['registration'] != key0 or teacher['external_outcomes_read'] or len(teacher['fits']) != 105
            or not held['before_held_scoring']):
        raise ValueError('Require frozen recipient chooser and unchanged held actions')
    paths = (Path(__file__), Path(model.__file__), Path(additive.__file__), Path(prior.model.__file__),
             Path(family.__file__), af, pf, OUT/'protocol.md')
    p = OUT/'registration.json'
    write(p, dict(files={str(x.resolve().relative_to(base.prior.ROOT)): sha(x) for x in paths},
        seeds=base.SEEDS, steps=prior.STEPS, batch=256, workers=0, threads=4,
        one_example_per_donor_recording=True, original_2pct=True, external_outcomes_read=False), immutable=True)
    return sha(p), source, bank, parent, teachers, teacher, held


@torch.inference_mode()
def examples(d, s, excluded, seed, bank, teachers, teacher, cut):
    reject = sorted([*excluded, s])
    if len(set(reject)) != 3:
        raise ValueError('Nomination and selection must exclude the donor')
    tag = nested.tag(reject, seed)
    ref = teacher['fits'][tag]['checkpoint']
    policy = base.load_net(LOCAL/f'{tag}_additive.pt', ref, action.study.model.PairValue(False), reject)
    ego = base.load_net(prior.LOCAL/f'{tag}_ego.pt', teachers['checkpoints']['ego'][tag],
                        action.model.AngleValue(False), reject)
    t = base.triple_id(reject, seed)
    forecaster = base.load_net(base.LOCAL/f'{t}.pt', bank['triple_checkpoints'][t],
                               nested.old.motion.motion.network(17), reject)
    offset, last = None, None
    x, y, counts = {'random': [], 'policy': []}, {'random': [], 'policy': []}, Counter()
    for m, q, a, anchor, _, _, target in nested.anchor_examples(d, s, reject, seed, forecaster):
        if q.start == last:
            continue
        last = q.start
        if offset is None:
            offset = screen.offsets(m['cv'], m['ols4'], target, cut)
        _, rec, ols0, cap = offset
        cv, ols = m['cv'][q], m['ols4'][q]
        legal = m['peak'][q] > 0.
        uniform, _, matched = family.match.align_anchor(anchor, a, m['paths'][q][a], legal[a])
        families = np.where(legal[a, 2], 2, matched.argmax(1))
        np.testing.assert_allclose(uniform[a], m['paths'][q][a, families], atol=1e-10, rtol=0.)
        raw = torch.from_numpy(m['x'][q])
        head = policy.head(policy.actor(raw)).numpy()
        b, n = model.best_pair(head, legal, families)
        selected = cv.copy()
        selected[b] = m['paths'][q][b, families]
        p, z = (np.linalg.norm(t[r]-cv[r], axis=-1).max(1) for t, r in ((uniform, a), (selected, b)))
        np.testing.assert_allclose(p, z, rtol=1e-10, atol=1e-10)
        counts.update(examples=1, legal_pairs=n, learned_slots=int((families == 2).sum()),
                      changed_actor_set=int(set(a) != set(b)))
        picks = []
        for mode, r, u in (('random', a, uniform), ('policy', b, selected)):
            f, _ = action.inputs(cv, u, r, d, q)
            (i, j), margin = screen.model.nominate(ego(torch.from_numpy(f)).numpy())
            k = rec[q.start]
            win = screen.oracle.legal_wins(cv, ols, target['future'][q], target['valid'][q], target['weight'][q],
                                          u, r, cut, ols0[k], cap[k])
            x[mode].append(prior.model.features(f, i, j, margin, q.stop-q.start))
            y[mode].append(float(win[i, j-1]))
            picks.append((int(r[i]), j))
        counts['changed_nominated_actor_angle'] += int(picks[0] != picks[1])
    if not x['random']:
        raise ValueError('No matched donor recordings')
    for a in y:
        counts[f'{a}_positive'] = int(sum(y[a]))
    return {a: (torch.from_numpy(np.stack(x[a])), torch.tensor(y[a], dtype=torch.float32),
                torch.full((len(y[a]),), base.SITES.index(s), dtype=torch.int64)) for a in x}, counts


def fit(identity, train, rank):
    saved = nested.old.LOCAL
    try:
        nested.old.LOCAL = LOCAL
        torch.manual_seed(identity['seed'])
        return nested.old.fit(identity, train, lambda m, b: prior.model.loss(m, b, rank),
                              prior.model.network(), prior.STEPS, 256)
    finally:
        nested.old.LOCAL = saved


@torch.inference_mode()
def propose(packet, d, old, nets):
    cv, anchor = packet['cv'], screen.pair.prediction(packet, 'additive')
    x = []
    for (lo, hi), a, (i, j), margin in zip(packet['bounds'], packet['additive_rows'],
            old['choices'], old['scores'][0], strict=True):
        q = slice(lo, hi)
        feats, _ = action.inputs(cv[q], anchor[q], a-lo, d, q)
        x.append(prior.model.features(feats, i, j, margin, hi-lo))
    z = torch.from_numpy(np.stack(x))
    return dict(choices=old['choices'], old_mask=old['old_mask'],
                scores=np.stack((old['scores'][0], *(n(z)[:, 0].numpy() for n in nets))))


def beat(phase, n, total):
    write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, completed=n, total=total))


def freeze(key, data, parent, bank, teachers, teacher, held):
    cells = {}
    for name, c in parent['cells'].items():
        p = LOCAL/f'{name}.json'
        excluded, seed = sorted(c['packets']), c['seed']
        donors, src = c['fit_sites'], held['cells'][name]
        if set(donors) != set(base.SITES)-set(excluded) or donors != src['fit_sites']:
            raise ValueError('Fit overlaps calibration or held location')
        if not p.exists():
            batches = [examples(data[s], s, excluded, seed, bank, teachers, teacher, c['cut']) for s in donors]
            counts = dict(sum((z for _, z in batches), Counter()))
            if counts['examples']*action.DRAWS != src['training']['examples']:
                raise ValueError('Expected the first of exactly three former draws')
            common = dict(registration=key, excluded=excluded, fit_sites=donors, seed=seed)
            fits = [fit(dict(name=f'{name}_{a}', **common), [z[a.split('_')[0]] for z, _ in batches],
                        a.endswith('budget')) for a in NEW]
            files, thresholds = {}, {}
            for s, r in c['packets'].items():
                h = src['packets'][s]
                with screen.packet_file(ctx.study.LOCAL, r) as pkt, screen.packet_file(family.LOCAL, h) as old:
                    v = propose(pkt, data[s], old, [t[0] for t in fits])
                    _, _, _, outcome, g = screen.observed(pkt, action.labels(data[s]['file']), c['cut'], v['choices'])
                    good = outcome['dual'] & (g[:, 0] > 0.)
                    thresholds[s] = {a: base.calibration(z, good) for a, z in zip(ARMS, v['scores'], strict=True)}
                    if tuple(thresholds[s]['margin']) != tuple(src['thresholds'][s]['margin']):
                        raise ValueError('Unchanged margin calibration did not replay')
                dest = LOCAL/f'{name}_{s}.npz'
                files[s] = dict(file=dest.name, sha256=base.parent.indices.save_npz(dest, v),
                                earlier=h['sha256'], original=r['sha256'])
            write(p, dict(registration=key, fit_sites=donors, training=counts,
                          checkpoints=[r for _, r in fits], packets=files, thresholds=thresholds), immutable=True)
        row = read(p)
        if (row['registration'] != key or any(sha(LOCAL/r['file']) != r['sha256'] for r in row['packets'].values())
                or any(sha(LOCAL/f"{r['identity']['name']}.pt") != r['sha256'] for r in row['checkpoints'])):
            raise ValueError('Changed frozen actor comparison')
        cells[name] = row
        beat('freeze', len(cells), 63)
        if len(cells) % 7 == 0:
            print('Frozen:', len(cells), '/ 63', flush=True)
    write(OUT/'freeze.json', dict(registration=key, before_held_scoring=True, cells=cells), immutable=True)


def evaluate(p, d, cut, f, rules):
    cv, ols, y, v, w = p['cv'], p['ols4'], d['future'], d['valid'], d['weight']
    picks = f['choices']
    anchor, changed, orig, edit, g = screen.observed(p, d, cut, picks)
    names, index, ols0, cap = screen.offsets(cv, ols, d, cut)
    strict, rel = [], []
    relative = ctx.prior.prior.pair_reduction
    for (lo, hi), a, (i, j), name in zip(p['bounds'], p['additive_rows'], picks, p['records'], strict=True):
        q = slice(lo, hi)
        k = index[lo]
        if names[k] != name or not np.all(index[q] == k):
            raise ValueError('Misaligned held query')
        wins = screen.oracle.legal_wins(cv[q], ols[q], y[q], v[q], w[q], anchor[q], a-lo, cut, ols0[k], cap[k])
        strict.append(wins[i, j-1])
        rel.append([relative(changed[None, q], anchor[q], y[q], v[q])[0, 1],
                    relative(anchor[None, q], changed[q], y[q], v[q])[0, 1]])
    strict, rel = np.array(strict, dtype=bool), np.asarray(rel)
    at = np.isin(d['recording'], p['records'])
    mass = w[at] @ np.where(v[at], np.linalg.norm(cv[at]-y[at], axis=-1), 0.).sum(1)
    n = len(picks)
    budgets = int(f['old_mask'].sum()), int(np.ceil(.15*n))
    chosen, scores = {}, {}
    for a, s in zip(ARMS, f['scores'], strict=True):
        lim = rules[a][0]
        chosen[a] = dict(zip(screen.QUOTAS, (*[screen.model.top_mask(s, k) for k in budgets],
                         s >= lim if lim is not None else np.zeros(n, dtype=bool)), strict=True))
        scores[a] = dict(auc_ap=screen.rank_metrics(s, strict))
        for q, use in chosen[a].items():
            scores[a][q] = dict(count=int(use.sum()), strict=int(strict[use].sum()),
                risk=int(edit['risk'][use].sum()), dual=int(edit['dual'][use].sum()),
                incremental=int((edit['dual'] & (g[:, 0] > 0.))[use].sum()),
                dual_all=int(np.where(use, edit['dual'], orig['dual']).sum()))
    if not np.array_equal(chosen['margin']['ego_count'], f['old_mask']):
        raise ValueError('Unchanged margin mask did not replay')
    paired = {}
    for a, b in PAIRS:
        for q in screen.QUOTAS:
            u, z = chosen[a][q], chosen[b][q]
            left, right = u & ~z, z & ~u
            paired[f'{a}_minus_{b}_{q}'] = dict(lower_pp=float(100.*(g[left, 0].sum()+g[right, 1].sum())/mass),
                relative=float((rel[left, 0].sum()+rel[right, 1].sum())/n),
                strict_pp=float(100.*(strict[u].sum()-strict[z].sum())/n))
    return dict(views=n, successful_nominees=int(strict.sum()), scores=scores, paired=paired)


def score(key, data, parent, held):
    p = OUT/'freeze.json'
    f = read(p)
    fr = sha(p)
    if f['registration'] != key or not f['before_held_scoring'] or set(f['cells']) != set(parent['cells']):
        raise ValueError('Require complete pre-score freeze')
    rows = []
    for name, c in parent['cells'].items():
        a = f['cells'][name]
        for s, r in c['packets'].items():
            cal, = set(c['packets'])-{s}
            z = a['packets'][s]
            if s in a['fit_sites'] or cal in a['fit_sites'] or z['original'] != r['sha256'] or (
                    z['earlier'] != held['cells'][name]['packets'][s]['sha256']):
                raise ValueError('Changed held nominee or source roles')
            dest = LOCAL/f'{name}_{s}_score.json'
            ident = dict(freeze=fr, site=s, calibration=cal, seed=c['seed'], packet=z['sha256'])
            if not dest.exists():
                with screen.packet_file(ctx.study.LOCAL, r) as q, screen.packet_file(LOCAL, z) as v:
                    value = evaluate(q, action.labels(data[s]['file']), c['cut'], v, a['thresholds'][cal])
                write(dest, dict(identity=ident, **value), immutable=True)
            value = read(dest)
            if value['identity'] != ident:
                raise ValueError('Changed held result')
            rows.append(value)
        beat('score', len(rows), 126)
    totals = {k: sum(r[k] for r in rows) for k in ('views', 'successful_nominees')}
    counts = {a: {q: dict(sum((Counter(r['scores'][a][q]) for r in rows), Counter()))
                  for q in screen.QUOTAS} for a in ARMS}
    old = read(family.OUT/'readout.json')
    if len(rows) != 126 or totals != old['totals'] or counts['margin'] != old['counts']['margin']:
        raise ValueError('Held roster or margin control changed')
    groups = [[r for r in rows if r['identity']['site'] == s] for s in base.SITES]
    draws = np.random.default_rng(2917).integers(7, size=(3000, 7))

    def ci(fn):
        def avg(seq):
            v = [z for z in seq if z is not None]
            return np.mean(v) if v else None
        x = np.asarray([avg(avg(fn(r) for r in g if r['identity']['seed'] == seed) for seed in base.SEEDS)
                        for g in groups], dtype=float)
        if not np.isfinite(x).all():
            return dict(estimable=False, sites=int(np.isfinite(x).sum()))
        return dict(point=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
                    by_site=dict(zip(base.SITES, x.tolist(), strict=True)))

    paired = {a: {k: ci(lambda r: r['paired'][a][k]) for k in d} for a, d in rows[0]['paired'].items()}
    ranking = {a: {k: ci(lambda r: (r['scores'][a]['auc_ap'] or {}).get(k)) for k in ('auc', 'ap')} for a in ARMS}
    stats = dict(sum((Counter(r['training']) for r in f['cells'].values()), Counter()))
    result = dict(result_source='fresh_run', registration=key, freeze=fr, totals=totals, counts=counts, paired=paired,
        training=stats, ranking=ranking, fits=252, updates=252*prior.STEPS, recordings=493,
        original_2pct=True, external_outcomes_read=False, independent_confirmation=False, deployment_changed=False)
    write(OUT/'readout.json', result, immutable=True)
    beat('complete', len(rows), 126)
    print(stats, counts, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze', 'score'))
    phase = p.parse_args().phase
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use native arm64 .venv-pytorch')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL/'screen.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, source, bank, parent, teachers, teacher, held = register()
        data = ctx.load_features(source)
        if phase == 'freeze':
            freeze(key, data, parent, bank, teachers, teacher, held)
        else:
            score(key, data, parent, held)
