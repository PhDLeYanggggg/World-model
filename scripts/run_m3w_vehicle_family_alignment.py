"""Compare family-aligned and uniform donor anchors at fixed held nominees."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_nominee_rank as prior
from src.world_model import m3w_vehicle_family_alignment as match

nested, screen, action, ctx, base, np, torch = (
    prior.nested, prior.screen, prior.action, prior.ctx, prior.base, prior.np, prior.torch)
sha, read, write = prior.sha, prior.read, prior.write
OUT = prior.OUT.parent/'opendd_vehicle_family_alignment_v1'
LOCAL = prior.LOCAL.parent/OUT.name
NEW = ('aligned_pointwise', 'aligned_budget')
ARMS = prior.ARMS+NEW
PAIRS = (('aligned_pointwise', 'pointwise'), ('aligned_budget', 'budget_rank'),
         ('aligned_pointwise', 'margin'), ('aligned_budget', 'margin'), ('aligned_budget', 'aligned_pointwise'))


def register():
    old, source, bank, parent, cap, upstream, trained = prior.register()
    pf = prior.OUT/'freeze.json'
    fixed = read(pf)
    ra = prior.OUT/'readout.json'
    a = read(ra)
    sh = prior.OUT/'anchor_shift.json'
    if (fixed['registration'] != old or not fixed['before_held_scoring'] or a['freeze'] != sha(pf)
            or a['external_outcomes_read'] or not read(sh)['zero_outcomes_materialized']):
        raise ValueError('Require frozen uniform-family fits and input-only audit')
    paths = (Path(__file__), Path(match.__file__), Path(prior.__file__), Path(prior.model.__file__),
             Path(nested.__file__), prior.OUT/'teacher_freeze.json', pf, ra, sh, OUT/'protocol.md')
    p = OUT/'registration.json'
    write(p, dict(files={str(f.resolve().relative_to(base.prior.ROOT)): sha(f) for f in paths},
        steps=prior.STEPS, batch=256, workers=0, threads=4, seeds=base.SEEDS, same_held_nominees=True,
        original_2pct=True, external_outcomes_read=False), immutable=True)
    return sha(p), source, bank, parent, fixed, trained


@torch.inference_mode()
def examples(d, site, excluded, seed, bank, teachers, cut):
    reject = sorted([*excluded, site])
    if len(set(reject)) != 3:
        raise ValueError('Expect two outer sites plus one donor')
    tag = nested.tag(reject, seed)
    ego = base.load_net(prior.LOCAL/f'{tag}_ego.pt', teachers['checkpoints']['ego'][tag],
                        action.model.AngleValue(False), reject)
    t = base.triple_id(reject, seed)
    forecast = base.load_net(base.LOCAL/f'{t}.pt', bank['triple_checkpoints'][t],
                             nested.old.motion.motion.network(17), reject)
    x0, x1, summaries, counts = [], [], [], Counter()
    offsets = None
    for m, q, rows, anchor, x, _, y in nested.anchor_examples(d, site, reject, seed, forecast):
        if offsets is None:
            offsets = screen.offsets(m['cv'], m['ols4'], y, cut)
        absrow = q.start+rows
        aligned, old, new = match.align_anchor(anchor, rows, m['paths'][absrow], m['peak'][absrow] > 0.)
        cv = m['cv'][q]
        p, v = (np.linalg.norm(a[rows]-cv[rows], axis=-1).max(1) for a in (anchor, aligned))
        np.testing.assert_allclose(p, v, rtol=1e-10, atol=1e-10)
        counts.update(examples=1, actor_slots=2, changed_anchors=int(not np.array_equal(anchor, aligned)),
            before_learned_mass=float((old[:, 2]/old.sum(1)).sum()),
            after_learned_mass=float((new[:, 2]/new.sum(1)).sum()))
        x0.append(x)
        x1.append(action.inputs(cv, aligned, rows, d, q)[0])
        summaries.append((m, q, rows, aligned))
    if not x1:
        raise ValueError('No aligned training examples')
    before, after = [ego(torch.from_numpy(np.stack(a))).numpy() for a in (x0, x1)]
    _, rec, ols0, cap = offsets
    features, labels = [], []
    for (m, q, rows, aligned), f, old, new in zip(summaries, x1, before, after, strict=True):
        (i, j), margin = screen.model.nominate(new)
        counts['changed_nominee_indices'] += int((i, j) != screen.model.nominate(old)[0])
        k = rec[q.start]
        wins = screen.oracle.legal_wins(m['cv'][q], m['ols4'][q], y['future'][q], y['valid'][q], y['weight'][q],
                                        aligned, rows, cut, ols0[k], cap[k])
        features.append(prior.model.features(f, i, j, margin, q.stop-q.start))
        labels.append(float(wins[i, j-1]))
    counts['positives'] = int(sum(labels))
    return (torch.from_numpy(np.stack(features)), torch.tensor(labels, dtype=torch.float32),
            torch.full((len(labels),), base.SITES.index(site), dtype=torch.int64)), counts


def fit(identity, data, rank):
    saved = nested.old.LOCAL
    try:
        nested.old.LOCAL = LOCAL
        torch.manual_seed(identity['seed'])
        return nested.old.fit(identity, data, lambda m, b: prior.model.loss(m, b, rank),
                              prior.model.network(), prior.STEPS, 256)
    finally:
        nested.old.LOCAL = saved


def freeze(key, data, parent, fixed, bank, teachers):
    cells = {}
    for name, c in parent['cells'].items():
        dest = LOCAL/f'{name}.json'
        excluded, seed = sorted(c['packets']), c['seed']
        donors = c['fit_sites']
        original = fixed['cells'][name]
        if set(donors) != set(base.SITES)-set(excluded) or donors != original['fit_sites']:
            raise ValueError('Changed train, validation or held source')
        if not dest.exists():
            batches = [examples(data[s], s, excluded, seed, bank, teachers, c['cut']) for s in donors]
            train = [x for x, _ in batches]
            stats = dict(sum((z for _, z in batches), Counter()))
            if stats['examples'] != original['n_train']:
                raise ValueError('Changed the number of donor draws')
            common = dict(registration=key, excluded=excluded, fit_sites=donors, seed=seed)
            fits = [fit(dict(name=f'{name}_{a}', **common), train, a == NEW[1]) for a in NEW]
            files, thresholds = {}, {}
            for s, r in c['packets'].items():
                src = original['packets'][s]
                with screen.packet_file(ctx.study.LOCAL, r) as p, screen.packet_file(prior.LOCAL, src) as z:
                    proposed = prior.propose(p, data[s], z, [m for m, _ in fits])
                    proposed['scores'] = np.concatenate((z['scores'], proposed['scores'][2:]), axis=0)
                    _, _, _, v, g = screen.observed(p, action.labels(data[s]['file']), c['cut'], proposed['choices'])
                    allowed = v['dual'] & (g[:, 0] > 0.)
                    thresholds[s] = {**original['threshold'][s], **{a: base.calibration(t, allowed)
                                      for a, t in zip(NEW, proposed['scores'][4:], strict=True)}}
                f = LOCAL/f'{name}_{s}.npz'
                files[s] = dict(file=f.name, sha256=base.parent.indices.save_npz(f, proposed),
                                prior=src['sha256'], original=r['sha256'])
            write(dest, dict(registration=key, training=stats, fit_sites=donors,
                checkpoints=[r for _, r in fits], packets=files, thresholds=thresholds), immutable=True)
        r = read(dest)
        if (r['registration'] != key or any(sha(LOCAL/v['file']) != v['sha256'] for v in r['packets'].values())
                or any(sha(LOCAL/f"{m['identity']['name']}.pt") != m['sha256'] for m in r['checkpoints'])):
            raise ValueError('Changed family-aligned packet or checkpoint')
        cells[name] = r
        beat('freeze', len(cells), 63)
        if len(cells) % 7 == 0:
            print('Frozen:', len(cells), '/ 63', flush=True)
    write(OUT/'freeze.json', dict(registration=key, before_held_scoring=True, cells=cells), immutable=True)


def beat(phase, n, total):
    write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, completed=n, total=total))


def evaluate(p, d, cut, f, rules):
    cv, ols, y, v, w = p['cv'], p['ols4'], d['future'], d['valid'], d['weight']
    picks = f['choices']
    anchor, changed, orig, edit, g = screen.observed(p, d, cut, picks)
    names, index, ols0, cap = screen.offsets(cv, ols, d, cut)
    strict, relative = [], []
    rel = ctx.prior.prior.pair_reduction
    for (lo, hi), a, (i, j), name in zip(p['bounds'], p['additive_rows'], picks, p['records'], strict=True):
        q = slice(lo, hi)
        k = index[lo]
        if names[k] != name or not np.all(index[q] == k):
            raise ValueError('Misaligned held query')
        wins = screen.oracle.legal_wins(cv[q], ols[q], y[q], v[q], w[q], anchor[q], a-lo, cut, ols0[k], cap[k])
        strict.append(wins[i, j-1])
        relative.append([rel(changed[None, q], anchor[q], y[q], v[q])[0, 1],
                         rel(anchor[None, q], changed[q], y[q], v[q])[0, 1]])
    strict, relative = np.array(strict, dtype=bool), np.asarray(relative)
    at = np.isin(d['recording'], p['records'])
    mass = w[at] @ np.where(v[at], np.linalg.norm(cv[at]-y[at], axis=-1), 0.).sum(1)
    n = len(picks)
    quotas = int(f['old_mask'].sum()), int(np.ceil(.15*n))
    chosen, scores = {}, {}
    for a, s in zip(ARMS, f['scores'], strict=True):
        lim = rules[a][0]
        chosen[a] = dict(zip(screen.QUOTAS, (*[screen.model.top_mask(s, k) for k in quotas],
                         s >= lim if lim is not None else np.zeros(n, dtype=bool)), strict=True))
        scores[a] = dict(auc_ap=screen.rank_metrics(s, strict))
        for q, take in chosen[a].items():
            scores[a][q] = dict(count=int(take.sum()), strict=int(strict[take].sum()),
                risk=int(edit['risk'][take].sum()), dual=int(edit['dual'][take].sum()),
                incremental=int((edit['dual'] & (g[:, 0] > 0.))[take].sum()),
                dual_all=int(np.where(take, edit['dual'], orig['dual']).sum()))
    if not np.array_equal(chosen['margin']['ego_count'], f['old_mask']):
        raise ValueError('Original margin selection did not replay')
    paired = {}
    for a, b in PAIRS:
        for q in screen.QUOTAS:
            u, z = chosen[a][q], chosen[b][q]
            left, right = u & ~z, z & ~u
            paired[f'{a}_minus_{b}_{q}'] = dict(lower_pp=float(100.*(g[left, 0].sum()+g[right, 1].sum())/mass),
                relative=float((relative[left, 0].sum()+relative[right, 1].sum())/n),
                strict_pp=float(100.*(strict[u].sum()-strict[z].sum())/n))
    return dict(views=n, successful_nominees=int(strict.sum()), scores=scores, paired=paired)


def score(key, data, parent, fixed):
    fp = OUT/'freeze.json'
    frozen = read(fp)
    fr = sha(fp)
    if frozen['registration'] != key or not frozen['before_held_scoring'] or set(frozen['cells']) != set(parent['cells']):
        raise ValueError('Require complete pre-score freeze')
    rows = []
    for name, c in parent['cells'].items():
        r = frozen['cells'][name]
        for s, p in c['packets'].items():
            cal, = set(c['packets'])-{s}
            z = r['packets'][s]
            if s in r['fit_sites'] or cal in r['fit_sites'] or z['original'] != p['sha256'] or (
                    z['prior'] != fixed['cells'][name]['packets'][s]['sha256']):
                raise ValueError('Changed held proposal or source roles')
            dest = LOCAL/f'{name}_{s}_score.json'
            ident = dict(freeze=fr, site=s, calibration=cal, seed=c['seed'], packet=z['sha256'])
            if not dest.exists():
                with screen.packet_file(ctx.study.LOCAL, p) as q, screen.packet_file(LOCAL, z) as held:
                    v = evaluate(q, action.labels(data[s]['file']), c['cut'], held, r['thresholds'][cal])
                write(dest, dict(identity=ident, **v), immutable=True)
            item = read(dest)
            if item['identity'] != ident:
                raise ValueError('Changed score receipt')
            rows.append(item)
        beat('score', len(rows), 126)
    totals = {k: sum(r[k] for r in rows) for k in ('views', 'successful_nominees')}
    counts = {a: {q: dict(sum((Counter(r['scores'][a][q]) for r in rows), Counter()))
                  for q in screen.QUOTAS} for a in ARMS}
    old = read(prior.OUT/'readout.json')
    if len(rows) != 126 or totals != old['totals'] or any(counts[a] != old['counts'][a] for a in prior.ARMS):
        raise ValueError('Uniform training controls changed')
    sites = [[r for r in rows if r['identity']['site'] == s] for s in base.SITES]
    draws = np.random.default_rng(2917).integers(7, size=(3000, 7))

    def ci(fn):
        def avg(seq):
            v = [z for z in seq if z is not None]
            return np.mean(v) if v else None
        x = np.asarray([avg(avg(fn(r) for r in g if r['identity']['seed'] == seed) for seed in base.SEEDS)
                        for g in sites], dtype=float)
        if not np.isfinite(x).all():
            return dict(estimable=False, sites=int(np.isfinite(x).sum()))
        return dict(point=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
                    by_site=dict(zip(base.SITES, x.tolist(), strict=True)))

    paired = {a: {k: ci(lambda r: r['paired'][a][k]) for k in v} for a, v in rows[0]['paired'].items()}
    ranking = {a: {k: ci(lambda r: (r['scores'][a]['auc_ap'] or {}).get(k)) for k in ('auc', 'ap')} for a in ARMS}
    stats = dict(sum((Counter(r['training']) for r in frozen['cells'].values()), Counter()))
    result = dict(result_source='fresh_run', registration=key, freeze=fr, totals=totals, counts=counts, paired=paired,
        training=stats, ranking=ranking, fits=126, updates=126*prior.STEPS, recordings=493,
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
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, source, bank, parent, fixed, teachers = register()
        data = ctx.load_features(source)
        if phase == 'freeze':
            freeze(key, data, parent, fixed, bank, teachers)
        else:
            score(key, data, parent, fixed)
