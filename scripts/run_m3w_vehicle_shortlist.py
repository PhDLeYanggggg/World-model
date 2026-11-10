"""Learn to select among three causal nominees and score matched interventions."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_policy_anchor as prior
from src.world_model import m3w_vehicle_shortlist as model

action, ctx, nested, screen, base, np, torch = (
    prior.action, prior.ctx, prior.nested, prior.screen, prior.base, prior.np, prior.torch)
read, write, sha = prior.read, prior.write, prior.sha
OUT = prior.OUT.parent/'opendd_vehicle_shortlist_v1'
LOCAL = prior.LOCAL.parent/OUT.name
ARMS = ('margin', 'ols_fixed', 'strict_fixed', 'utility_fixed', 'strict_top1', 'utility_top1', 'strict_top3', 'utility_top3')
PAIRS = (('ols_fixed', 'margin'), ('strict_fixed', 'margin'), ('utility_fixed', 'margin'),
         ('strict_top1', 'margin'), ('utility_top1', 'margin'), ('strict_top3', 'strict_top1'),
         ('utility_top3', 'utility_top1'), ('strict_top3', 'margin'), ('utility_top3', 'margin'))
PAIR = ctx.prior.prior.pair_reduction


def register():
    pre, source, bank, parent, teachers, chooser, previous = prior.register()
    f, r, cap = (prior.OUT/k for k in ('freeze.json', 'readout.json', 'topk_capacity.json'))
    fixed = read(f)
    e = read(r)
    if fixed['registration'] != pre or e['freeze'] != sha(f) or e['external_outcomes_read'] or read(cap)['summary']['hits'][:2] != [420, 802]:
        raise ValueError('Require unchanged top-one and its verified shortlist')
    p = OUT/'registration.json'
    paths = (Path(__file__), Path(model.__file__), Path(prior.__file__), Path(prior.prior.model.__file__),
             f, r, cap, prior.OUT/'additive_freeze.json', OUT/'protocol.md')
    write(p, dict(files={str(q.resolve().relative_to(base.prior.ROOT)): sha(q) for q in paths},
        seeds=base.SEEDS, shortlist=3, steps=prior.prior.STEPS, batch=256, workers=0, threads=4,
        original_2pct=True, external_outcomes_read=False), immutable=True)
    return sha(p), source, bank, parent, teachers, chooser, fixed


def options(cv, anchor, sel, shortlist):
    bank = action.model.candidates(cv[sel], anchor[sel])
    p = np.repeat(anchor[None], len(shortlist)+1, axis=0)
    for k, (i, j) in enumerate(shortlist, 1):
        p[k, sel[i]] = bank[i, j]
    return p


def outcomes(cv, ols, anchor, paths, sel, y, v, w, cut, offset, cap):
    """Exact original risk and utility checks for all four paths at the query."""
    rows = np.tile(sel, len(paths))
    z = paths[:, sel].reshape(-1, *cv.shape[1:])
    m = screen.oracle.moments(cv[rows], z, y[rows], v[rows], cut[0])
    risk = np.column_stack([m[k] for k in ('all_excess', 'easy_excess', 'easy_signed')]).reshape(-1, 2, 3).sum(1)
    safe = (risk <= [0., 0., cap]).all(1) & (m['ref_easy'].reshape(-1, 2).sum(1) > 0.)
    pair = base.parent.paired

    def per(a, b):
        return (pair(a, b, y[rows], v[rows])[1].sum(1)*w[rows]).reshape(-1, 2).sum(1)

    dual = safe & (per(cv[rows], z) > 0.) & (per(ols[rows], z)-per(ols[rows], cv[rows])+offset > 0.)
    gain = per(anchor[rows], z)
    rel = PAIR(paths, anchor, y, v)[:, 1]
    strict = dual & (gain >= -1e-10) & (rel > 1e-10)
    release = dual & (gain > 0.)
    return safe, dual, release, strict, rel


def feature_rows(x, shortlist, margin, n):
    return np.stack([prior.prior.model.features(x, i, j, s, n)
                     for (i, j), s in zip(shortlist, margin, strict=True)])


def labels(anchor, paths, sel, shortlist, y, v, out):
    safe, _, release, strict, rel = out
    r = sel[shortlist[:, 0]]
    before, after = anchor[r], paths[1:, r][np.arange(3), np.arange(3)]
    benefit, harm, shift = action.model.gain_harm(before, after, y[r], v[r])
    d = shift.sum(1).clip(1e-9)
    envelope = (d/anchor.shape[1])*2./len(anchor)
    return np.column_stack((strict[1:], safe[1:], release[1:], benefit.sum(1)/d, harm.sum(1)/d,
                            (rel[1:]/envelope).clip(-1., 1.))).astype(np.float32)


@torch.inference_mode()
def donor(d, s, excluded, seed, bank, teachers, chooser, cut):
    ex = sorted([*excluded, s])
    if len(set(ex)) != 3:
        raise ValueError('Require source-excluded donor proposal models')
    tag = nested.tag(ex, seed)
    e = base.load_net(prior.prior.LOCAL/f'{tag}_ego.pt', teachers['checkpoints']['ego'][tag],
                      action.model.AngleValue(False), ex)
    a = base.load_net(prior.LOCAL/f'{tag}_additive.pt', chooser['fits'][tag]['checkpoint'],
                      action.study.model.PairValue(False), ex)
    t = base.triple_id(ex, seed)
    forecaster = base.load_net(base.LOCAL/f'{t}.pt', bank['triple_checkpoints'][t],
                               nested.old.motion.motion.network(17), ex)
    target = action.labels(d['file'])
    m = action.study.menu(nested.data_slice(d, target), forecaster)
    _, rec, off, cap = screen.offsets(m['cv'], m['ols4'], target, cut)
    x, z = [], []
    for lo, hi in m['slots'].values():
        q = slice(lo, hi)
        rows, families, i, j = action.study.choices(m['peak'], lo, hi)
        h = a.head(a.actor(torch.from_numpy(m['x'][rows, families]))).numpy()
        k = int(action.study.model.merit(.5*(h[i]+h[j])).argmax())
        pick = [i[k], j[k]]
        sel = rows[pick]-lo
        cv, ols = m['cv'][q], m['ols4'][q]
        anchor = cv.copy()
        anchor[sel] = m['paths'][rows[pick], families[pick]]
        feats, _ = action.inputs(cv, anchor, sel, d, q)
        short, margin = model.order(e(torch.from_numpy(feats)).numpy())
        paths = options(cv, anchor, sel, short)
        y, v, w = (target[k][q] for k in ('future', 'valid', 'weight'))
        out = outcomes(cv, ols, anchor, paths, sel, y, v, w, cut, off[rec[lo]], cap[rec[lo]])
        z.append(labels(anchor, paths, sel, short, y, v, out))
        x.append(feature_rows(feats, short, margin, hi-lo))
    if not x:
        raise ValueError('No source-excluded shortlist training samples')
    return tuple(torch.from_numpy(np.concatenate(a)) for a in (x, z))


def fit(identity, data, utility):
    saved = nested.old.LOCAL
    try:
        nested.old.LOCAL = LOCAL
        torch.manual_seed(identity['seed'])
        return nested.old.fit(identity, data, lambda m, b: model.loss(m, b, utility), model.network(),
                              prior.prior.STEPS, 256)
    finally:
        nested.old.LOCAL = saved


@torch.inference_mode()
def proposals(packet, d, archived, e, nets):
    cv, ols = packet['cv'], packet['ols4']
    anchor = screen.pair.prediction(packet, 'additive')
    features, shortlists, envelope, ols_score = [], [], [], []
    for (lo, hi), rows, old in zip(packet['bounds'], packet['additive_rows'], archived['choices'], strict=True):
        q = slice(lo, hi)
        sel = rows-lo
        f, _ = action.inputs(cv[q], anchor[q], sel, d, q)
        short, margin = model.order(e(torch.from_numpy(f)).numpy())
        np.testing.assert_array_equal(short[0], old)
        features.append(feature_rows(f, short, margin, hi-lo))
        shortlists.append(short)
        p = options(cv[q], anchor[q], sel, short)
        envelope.append(np.linalg.norm(p[1:]-p[0], axis=-1).sum((1, 2))/p.shape[2])
        ols_score.append(-np.linalg.norm(p[1:]-ols[None, q], axis=-1).sum((1, 2)))
    x = torch.from_numpy(np.stack(features))
    u = np.asarray(envelope)
    strict, _ = model.scores(nets[0](x).numpy(), u)
    _, utility = model.scores(nets[1](x).numpy(), u)
    margin = archived['scores'][0]
    zeros = np.zeros(len(x), dtype=int)
    a, b, c = (z.argmax(1) for z in (np.asarray(ols_score), strict, utility))
    choose = np.stack((zeros, a, b, c, zeros, zeros, b, c))
    priority = np.stack((margin, margin, margin, margin, strict[:, 0], utility[:, 0], strict.max(1), utility.max(1)))
    return dict(shortlist=np.stack(shortlists), choices=choose, scores=priority, old_mask=archived['old_mask'])


def truth(packet, d, cut, shortlisted):
    cv, ols = packet['cv'], packet['ols4']
    anchor = screen.pair.prediction(packet, 'additive')
    names, rec, off, cap = screen.offsets(cv, ols, d, cut)
    for (lo, hi), rows, short, name in zip(packet['bounds'], packet['additive_rows'], shortlisted, packet['records'], strict=True):
        q = slice(lo, hi)
        i = rec[lo]
        if names[i] != name or not np.all(rec[q] == i):
            raise ValueError('Wrong frozen query')
        y, v, w = (d[k][q] for k in ('future', 'valid', 'weight'))
        sel = rows-lo
        paths = options(cv[q], anchor[q], sel, short)
        yield paths, y, v, w, outcomes(cv[q], ols[q], anchor[q], paths, sel, y, v, w, cut, off[i], cap[i])


def beat(phase, n, total):
    write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, completed=n, total=total))


def freeze(key, data, bank, parent, teachers, chooser, fixed):
    old_action = read(action.OUT/'freeze.json')
    cells = {}
    for name, c in parent['cells'].items():
        p = LOCAL/f'{name}.json'
        excluded, seed = sorted(c['packets']), c['seed']
        donors, src = c['fit_sites'], fixed['cells'][name]
        if set(donors) != set(base.SITES)-set(excluded):
            raise ValueError('Changed training roles')
        if not p.exists():
            train = [donor(data[s], s, excluded, seed, bank, teachers, chooser, c['cut']) for s in donors]
            if sum(len(y) for _, y in train) != 3*src['training']['examples']:
                raise ValueError('Changed donor recording count')
            common = dict(registration=key, excluded=excluded, fit_sites=donors, seed=seed)
            fits = [fit(dict(name=f'{name}_{a}', **common), train, a == 'utility') for a in ('strict', 'utility')]
            r = old_action['cells'][name]['checkpoints']['ego']
            e = base.load_net(action.LOCAL/f"{r['identity']['name']}.pt", r, action.model.AngleValue(False), excluded)
            files, thresholds = {}, {}
            for s, original in c['packets'].items():
                f = src['packets'][s]
                with screen.packet_file(ctx.study.LOCAL, original) as p0, screen.packet_file(prior.LOCAL, f) as archived:
                    z = proposals(p0, data[s], archived, e, [m for m, _ in fits])
                    ready = np.asarray([row[4][2][1:] for row in truth(p0, action.labels(data[s]['file']), c['cut'], z['shortlist'])])
                    thresholds[s] = {a: base.calibration(v, ready[np.arange(len(ready)), i])
                                     for a, v, i in zip(ARMS, z['scores'], z['choices'], strict=True)}
                    if tuple(thresholds[s]['margin']) != tuple(src['thresholds'][s]['margin']):
                        raise ValueError('Frozen margin calibration changed')
                dest = LOCAL/f'{name}_{s}.npz'
                files[s] = dict(file=dest.name, sha256=base.parent.indices.save_npz(dest, z),
                                original=original['sha256'], prior=f['sha256'])
            y = torch.cat([y for _, y in train])
            write(p, dict(registration=key, fit_sites=donors, training=dict(rows=len(y), positives=int(y[:, 0].sum())),
                          checkpoints=[r for _, r in fits], packets=files, thresholds=thresholds), immutable=True)
        row = read(p)
        if (row['registration'] != key or any(sha(LOCAL/r['file']) != r['sha256'] for r in row['packets'].values())
                or any(sha(LOCAL/f"{r['identity']['name']}.pt") != r['sha256'] for r in row['checkpoints'])):
            raise ValueError('Changed shortlist checkpoint or scores')
        cells[name] = row
        beat('freeze', len(cells), 63)
        if len(cells) % 7 == 0:
            print('Frozen:', len(cells), '/ 63', flush=True)
    write(OUT/'freeze.json', dict(registration=key, before_held_scoring=True, cells=cells), immutable=True)


def evaluate(p, d, cut, f, rules):
    n = len(f['shortlist'])
    sizes = (int(f['old_mask'].sum()), int(np.ceil(.15*n)))
    chosen = {}
    for a, scores, i in zip(ARMS, f['scores'], f['choices'], strict=True):
        t = rules[a][0]
        masks = (*[screen.model.top_mask(scores, k) for k in sizes], scores >= t if t is not None else np.zeros(n, dtype=bool))
        chosen[a] = dict(zip(screen.QUOTAS, (np.where(m, i+1, 0) for m in masks), strict=True))
    if not np.array_equal(chosen['margin']['ego_count'] > 0, f['old_mask']):
        raise ValueError('Unchanged margin mask did not replay')
    counts = {a: {q: Counter() for q in screen.QUOTAS} for a in ARMS}
    paired = {f'{a}_minus_{b}_{q}': np.zeros(4) for a, b in PAIRS for q in screen.QUOTAS}
    strict_top1 = strict_top3 = 0
    for k, (paths, y, v, w, out) in enumerate(truth(p, d, cut, f['shortlist'])):
        safe, dual, release, strict, _ = out
        strict_top1 += int(strict[1])
        strict_top3 += int(strict[1:].any())
        fde = np.where(v[:, -1], np.linalg.norm(paths[:, :, -1]-y[:, -1], axis=-1), 0.) @ w
        for a, z in chosen.items():
            for q, i in z.items():
                j = i[k]
                counts[a][q].update(count=int(j > 0), strict=int(strict[j]), risk=int(j > 0 and safe[j]),
                    dual=int(j > 0 and dual[j]), incremental=int(release[j]), dual_all=int(dual[j]))
        for a, b in PAIRS:
            for q in screen.QUOTAS:
                i, j = chosen[a][q][k], chosen[b][q][k]
                if i != j:
                    gain = w @ base.parent.paired(paths[j], paths[i], y, v)[1].sum(1)
                    pair = PAIR(paths[i][None], paths[j], y, v)[0, 1]
                    paired[f'{a}_minus_{b}_{q}'] += gain, pair, int(strict[i])-int(strict[j]), fde[j]-fde[i]
    at = np.isin(d['recording'], p['records'])
    cv, y, v, w = p['cv'][at], d['future'][at], d['valid'][at], d['weight'][at]
    err = np.where(v, np.linalg.norm(cv-y, axis=-1), 0.)
    mass, last = w @ err.sum(1), w @ err[:, -1]
    for k, z in paired.items():
        paired[k] = dict(lower_pp=float(100.*z[0]/mass), relative=float(z[1]/n), strict_pp=float(100.*z[2]/n),
                         observed_fde_pp=float(100.*z[3]/last) if last > 0. else None)
    return dict(views=n, strict_top1=strict_top1, strict_top3=strict_top3, counts=counts, paired=paired)


def score(key, data, parent, fixed):
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
                    z['prior'] != fixed['cells'][name]['packets'][s]['sha256']):
                raise ValueError('Changed shortlist source roles or candidate')
            dest = LOCAL/f'{name}_{s}_score.json'
            ident = dict(freeze=fr, site=s, seed=c['seed'], calibration=cal, packet=z['sha256'])
            if not dest.exists():
                with screen.packet_file(ctx.study.LOCAL, r) as p0, screen.packet_file(LOCAL, z) as v:
                    item = evaluate(p0, action.labels(data[s]['file']), c['cut'], v, a['thresholds'][cal])
                write(dest, dict(identity=ident, **item), immutable=True)
            item = read(dest)
            if item['identity'] != ident:
                raise ValueError('Changed shortlist score')
            rows.append(item)
        beat('score', len(rows), 126)
    totals = {k: sum(r[k] for r in rows) for k in ('views', 'strict_top1', 'strict_top3')}
    counts = {a: {q: dict(sum((Counter(r['counts'][a][q]) for r in rows), Counter()))
                  for q in screen.QUOTAS} for a in ARMS}
    if len(rows) != 126 or totals != dict(views=8874, strict_top1=420, strict_top3=802) or (
            counts['margin'] != read(prior.OUT/'readout.json')['counts']['margin']):
        raise ValueError('Fixed proposal ceiling or margin control did not replay')
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
    result = dict(result_source='fresh_run', registration=key, freeze=fr, totals=totals, counts=counts, paired=paired,
        training=dict(sum((Counter(r['training']) for r in f['cells'].values()), Counter())),
        fits=126, updates=126*prior.prior.STEPS, recordings=493, original_2pct=True, external_outcomes_read=False,
        independent_confirmation=False, deployment_changed=False)
    write(OUT/'readout.json', result, immutable=True)
    beat('complete', len(rows), 126)
    print(totals, counts, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze', 'score'))
    phase = p.parse_args().phase
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use native arm64 .venv-pytorch')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, source, bank, parent, teachers, chooser, fixed = register()
        data = ctx.load_features(source)
        if phase == 'freeze':
            freeze(key, data, bank, parent, teachers, chooser, fixed)
        else:
            score(key, data, parent, fixed)
