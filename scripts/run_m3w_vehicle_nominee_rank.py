"""Cross-fit exact nominee outcomes, freeze, and score equal-budget choices."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import fit_m3w_vehicle_nested_nominees as nested
from src.world_model import m3w_vehicle_nominee_rank as model

screen, action, ctx, base, np, torch = nested.screen, nested.action, nested.ctx, nested.base, nested.np, nested.torch
OUT, LOCAL, sha, read, write = nested.OUT, nested.LOCAL, nested.sha, nested.read, nested.write
ARMS = ('margin', 'opportunity', 'pointwise', 'budget_rank')
PAIRS = (('pointwise', 'margin'), ('budget_rank', 'margin'), ('budget_rank', 'pointwise'), ('budget_rank', 'opportunity'))
STEPS = 200


def register():
    teacher, source, forecast, parent, old_ego, cap, old_screen = nested.manifest()
    fp = OUT/'teacher_freeze.json'
    trained = read(fp)
    if (trained['registration'] != teacher or trained['external_outcomes_read']
            or any(len(trained['checkpoints'][k]) != 105 for k in ('ego', 'forecast'))):
        raise ValueError('Require the complete source-excluded teacher bank')
    files = (Path(__file__), Path(model.__file__), Path(nested.__file__), Path(screen.__file__),
             fp, screen.OUT/'readout.json', OUT/'protocol.md')
    p = OUT/'registration.json'
    write(p, dict(files={str(f.resolve().relative_to(base.prior.ROOT)): sha(f) for f in files}, seeds=base.SEEDS,
        steps=STEPS, batch=256, workers=0, threads=4, same_nominee=True, original_2pct=True,
        external_outcomes_read=False), immutable=True)
    return sha(p), source, forecast, parent, cap, old_screen, trained


@torch.inference_mode()
def examples(d, site, excluded, seed, bank, trained, cut):
    reject = sorted([*excluded, site])
    if len(set(reject)) != 3:
        raise ValueError('Nomination needs a distinct, excluded donor')
    tag = nested.tag(reject, seed)
    r = trained['checkpoints']['ego'][tag]
    ego = base.load_net(LOCAL/f'{tag}_ego.pt', r, action.model.AngleValue(False), reject)
    t = base.triple_id(reject, seed)
    forecast = base.load_net(base.LOCAL/f'{t}.pt', bank['triple_checkpoints'][t],
                             nested.old.motion.motion.network(17), reject)
    summary, feats, y = [], [], []
    offsets = None
    for m, q, a, anchor, x, angles, target in nested.anchor_examples(d, site, reject, seed, forecast):
        if offsets is None:
            offsets = screen.offsets(m['cv'], m['ols4'], target, cut)
        _, rec, ols0, cap = offsets
        # Predict every angle from an excluded donor model before deriving the outcome.
        summary.append((m, q, a, anchor))
        feats.append(x)
    pred = ego(torch.from_numpy(np.stack(feats))).numpy()
    features = []
    for (m, q, a, anchor), x, v in zip(summary, feats, pred, strict=True):
        (i, j), margin = screen.model.nominate(v)
        k = rec[q.start]
        win = screen.oracle.legal_wins(m['cv'][q], m['ols4'][q], target['future'][q],
            target['valid'][q], target['weight'][q], anchor, a, cut, ols0[k], cap[k])
        features.append(model.features(x, i, j, margin, q.stop-q.start))
        y.append(float(win[i, j-1]))
    if not features:
        raise ValueError('No nominated donor examples')
    return torch.from_numpy(np.stack(features)), torch.tensor(y, dtype=torch.float32), torch.full(
        (len(y),), base.SITES.index(site), dtype=torch.int64)


def fit(identity, train, rank):
    saved = nested.old.LOCAL
    try:
        nested.old.LOCAL = LOCAL
        torch.manual_seed(identity['seed'])
        return nested.old.fit(identity, train, lambda m, b: model.loss(m, b, rank), model.network(), STEPS, 256)
    finally:
        nested.old.LOCAL = saved


@torch.inference_mode()
def propose(packet, d, old, models):
    cv, anchor = packet['cv'], screen.pair.prediction(packet, 'additive')
    batch = []
    for (lo, hi), a, (i, j), margin in zip(packet['bounds'], packet['additive_rows'], old['choices'],
                                         old['scores'][0], strict=True):
        q = slice(lo, hi)
        x, _ = action.inputs(cv[q], anchor[q], a-lo, d, q)
        batch.append(model.features(x, i, j, margin, hi-lo))
    batch = torch.from_numpy(np.stack(batch))
    return dict(choices=old['choices'], old_mask=old['old_mask'],
        scores=np.stack((*old['scores'][:2], *(m(batch)[:, 0].numpy() for m in models))))


def freeze(key, data, parent, old_screen, bank, trained):
    cells = {}
    for name, c in parent['cells'].items():
        fp = LOCAL/f'{name}.json'
        excluded, seed = sorted(c['packets']), c['seed']
        donors = c['fit_sites']
        if set(donors) != set(base.SITES)-set(excluded):
            raise ValueError('Overlapping calibration or held location')
        if not fp.exists():
            train = [examples(data[s], s, excluded, seed, bank, trained, c['cut']) for s in donors]
            common = dict(registration=key, excluded=excluded, fit_sites=donors, seed=seed)
            fits = [fit(dict(name=f'{name}_{a}', **common), train, a == 'budget_rank') for a in ARMS[2:]]
            records, thresholds = {}, {}
            src = old_screen['cells'][name]
            for s, r in c['packets'].items():
                with screen.packet_file(ctx.study.LOCAL, r) as p, screen.packet_file(screen.LOCAL, src['packets'][s]) as old:
                    v = propose(p, data[s], old, [t[0] for t in fits])
                    _, _, _, eval_, g = screen.observed(p, action.labels(data[s]['file']), c['cut'], v['choices'])
                    passes = eval_['dual'] & (g[:, 0] > 0.)
                    thresholds[s] = {a: base.calibration(z, passes) for a, z in zip(ARMS, v['scores'], strict=True)}
                for a in ARMS[:2]:
                    if tuple(thresholds[s][a]) != tuple(src['thresholds_from_calibration_site'][s][a]):
                        raise ValueError('Frozen control calibration changed')
                p = LOCAL/f'{name}_{s}.npz'
                records[s] = dict(file=p.name, sha256=base.parent.indices.save_npz(p, v), original=r['sha256'],
                                  frozen_nominees=src['packets'][s]['sha256'])
            write(fp, dict(registration=key, fit_sites=donors, n_train=sum(len(y) for _, y, _ in train),
                n_positive=int(sum(y.sum().item() for _, y, _ in train)), checkpoint=[t[1] for t in fits],
                packets=records, threshold=thresholds), immutable=True)
        r = read(fp)
        if (r['registration'] != key or any(sha(LOCAL/z['file']) != z['sha256'] for z in r['packets'].values())
                or any(sha(LOCAL/f"{m['identity']['name']}.pt") != m['sha256'] for m in r['checkpoint'])):
            raise ValueError('Changed nominee ranker freeze')
        cells[name] = r
        beat('freeze', len(cells), 63)
        if len(cells) % 7 == 0:
            print('Frozen:', len(cells), '/ 63', flush=True)
    write(OUT/'freeze.json', dict(registration=key, before_held_scoring=True, cells=cells), immutable=True)


def beat(phase, n, total):
    write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, completed=n, total=total))


def evaluate(p, d, cut, f, thresholds):
    cv, ols, y, v, w = p['cv'], p['ols4'], d['future'], d['valid'], d['weight']
    picks, score = f['choices'], f['scores']
    anchor, changed, orig, edit, g = screen.observed(p, d, cut, picks)
    names, index, ols0, cap = screen.offsets(cv, ols, d, cut)
    strict, relative = [], []
    rel = ctx.prior.prior.pair_reduction
    for (lo, hi), a, (i, j), name in zip(p['bounds'], p['additive_rows'], picks, p['records'], strict=True):
        q = slice(lo, hi)
        k = index[lo]
        if names[k] != name or not np.all(index[q] == k):
            raise ValueError('Misaligned evaluation query')
        win = screen.oracle.legal_wins(cv[q], ols[q], y[q], v[q], w[q], anchor[q], a-lo, cut, ols0[k], cap[k])
        strict.append(win[i, j-1])
        relative.append([rel(changed[None, q], anchor[q], y[q], v[q])[0, 1],
                         rel(anchor[None, q], changed[q], y[q], v[q])[0, 1]])
    strict, relative = np.array(strict, dtype=bool), np.asarray(relative)
    at = np.isin(d['recording'], p['records'])
    mass = w[at] @ np.where(v[at], np.linalg.norm(cv[at]-y[at], axis=-1), 0.).sum(1)
    n = len(picks)
    quotas = int(f['old_mask'].sum()), int(np.ceil(.15*n))
    chosen, scores = {}, {}
    for a, s in zip(ARMS, score, strict=True):
        lim = thresholds[a][0]
        chosen[a] = dict(zip(screen.QUOTAS, (*[screen.model.top_mask(s, k) for k in quotas],
                                   s >= lim if lim is not None else np.zeros(n, dtype=bool)), strict=True))
        scores[a] = dict(auc_ap=screen.rank_metrics(s, strict))
        for q, take in chosen[a].items():
            scores[a][q] = dict(count=int(take.sum()), strict=int(strict[take].sum()),
                risk=int(edit['risk'][take].sum()), dual=int(edit['dual'][take].sum()),
                incremental=int((edit['dual'] & (g[:, 0] > 0.))[take].sum()),
                dual_all=int(np.where(take, edit['dual'], orig['dual']).sum()))
    if not np.array_equal(chosen['margin']['ego_count'], f['old_mask']):
        raise ValueError('Original action-margin control does not replay')
    paired = {}
    for a, b in PAIRS:
        for q in screen.QUOTAS:
            u, z = chosen[a][q], chosen[b][q]
            left, right = u & ~z, z & ~u
            paired[f'{a}_minus_{b}_{q}'] = dict(lower_pp=float(100.*(g[left, 0].sum()+g[right, 1].sum())/mass),
                relative=float((relative[left, 0].sum()+relative[right, 1].sum())/n),
                strict_pp=float(100.*(strict[u].sum()-strict[z].sum())/n))
    return dict(views=n, successful_nominees=int(strict.sum()), scores=scores, paired=paired)


def score(key, data, parent):
    fp = OUT/'freeze.json'
    frozen = read(fp)
    fr = sha(fp)
    if frozen['registration'] != key or not frozen['before_held_scoring'] or set(frozen['cells']) != set(parent['cells']):
        raise ValueError('Require pre-score freeze')
    rows = []
    for name, c in parent['cells'].items():
        r = frozen['cells'][name]
        for s, p in c['packets'].items():
            cal, = set(c['packets'])-{s}
            if s in r['fit_sites'] or cal in r['fit_sites'] or r['packets'][s]['original'] != p['sha256']:
                raise ValueError('Changed held or calibration roles')
            dest = LOCAL/f'{name}_{s}_score.json'
            ident = dict(freeze=fr, site=s, calibration=cal, seed=c['seed'], packet=r['packets'][s]['sha256'])
            if not dest.exists():
                with screen.packet_file(ctx.study.LOCAL, p) as pkt, screen.packet_file(LOCAL, r['packets'][s]) as a:
                    values = evaluate(pkt, action.labels(data[s]['file']), c['cut'], a, r['threshold'][cal])
                write(dest, dict(identity=ident, **values), immutable=True)
            item = read(dest)
            if item['identity'] != ident:
                raise ValueError('Changed evaluation receipt')
            rows.append(item)
        beat('score', len(rows), 126)
    totals = {k: sum(r[k] for r in rows) for k in ('views', 'successful_nominees')}
    counts = {a: {q: dict(sum((Counter(r['scores'][a][q]) for r in rows), Counter()))
                  for q in screen.QUOTAS} for a in ARMS}
    prev = read(screen.OUT/'readout.json')
    mapping = dict(count='count', strict='strict', risk='actual_risk', dual='actual_dual',
                   incremental='positive_incremental', dual_all='dual_all')
    if totals != dict(views=8874, successful_nominees=420) or any(
            counts[a][q].get(k, 0) != prev['counts'][a][q].get(v, 0) for a in ARMS[:2] for q in screen.QUOTAS
            for k, v in mapping.items()):
        raise ValueError('Frozen nominees or control results changed')
    sites = [[r for r in rows if r['identity']['site'] == s] for s in base.SITES]
    draws = np.random.default_rng(2917).integers(7, size=(3000, 7))

    def ci(fn):
        def avg(seq):
            v = [z for z in seq if z is not None]
            return np.mean(v) if v else None
        v = np.asarray([avg(avg(fn(r) for r in g if r['identity']['seed'] == seed) for seed in base.SEEDS)
                        for g in sites], dtype=float)
        if not np.isfinite(v).all():
            return dict(estimable=False, sites=int(np.isfinite(v).sum()))
        return dict(point=float(v.mean()), ci95=np.quantile(v[draws].mean(1), [.025, .975]).tolist(),
                    by_site=dict(zip(base.SITES, v.tolist(), strict=True)))

    ranking = {a: {k: ci(lambda r: (r['scores'][a]['auc_ap'] or {}).get(k)) for k in ('auc', 'ap')} for a in ARMS}
    paired = {a: {k: ci(lambda r: r['paired'][a][k]) for k in v} for a, v in rows[0]['paired'].items()}
    result = dict(result_source='fresh_run', registration=key, freeze=fr, totals=totals, counts=counts,
        ranking=ranking, paired=paired, fits=126, updates=126*STEPS, recordings=493,
        training_examples=sum(r['n_train'] for r in frozen['cells'].values()),
        training_positives=sum(r['n_positive'] for r in frozen['cells'].values()),
        original_2pct=True, external_outcomes_read=False, independent_confirmation=False, deployment_changed=False)
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
    with (LOCAL/'ranker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, source, bank, upstream, cap, old_screen, teachers = register()
        data = ctx.load_features(source)
        if phase == 'freeze':
            freeze(key, data, upstream, old_screen, bank, teachers)
        else:
            score(key, data, upstream)
