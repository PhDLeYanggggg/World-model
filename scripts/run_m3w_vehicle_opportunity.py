"""Freeze a recording opportunity head and compare screens with identical nominees."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy.special import expit
from scipy.stats import rankdata
from scripts import audit_m3w_vehicle_action_capture as oracle
from scripts import run_m3w_vehicle_action_value as action
from src.world_model import m3w_vehicle_opportunity as model

ctx, np, torch, base, pair = action.context, action.np, action.torch, action.base, action.pair
old, sha, read, write = action.old, action.sha, action.read, action.write
OUT = action.OUT.parent/'opendd_vehicle_opportunity_v1'
LOCAL = action.LOCAL.parent/OUT.name
ARMS = ('margin', 'opportunity', 'product')
QUOTAS = ('ego_count', 'fixed_15pct', 'calibrated')
STEPS = 200
packet_file = ctx.prior.prior.ref.prior.packet


def register():
    parent, source, forecasts, f, _ = action.register()
    frozen_path, capture_path = (action.OUT/k for k in ('freeze.json', 'capture_readout.json'))
    fixed = read(frozen_path)
    cap = read(capture_path)
    if (fixed['registration'] != parent or not fixed['freeze_before_held_score']
            or cap['external_outcomes_read'] or cap['strict_one_rotation_capacity'] != 1393):
        raise ValueError('Require frozen action choices and original finite-menu support')
    paths = (Path(__file__), Path(model.__file__), Path(oracle.__file__), Path(action.__file__),
             frozen_path, capture_path, OUT/'protocol.md')
    p = OUT/'registration.json'
    write(p, dict(files={str(x.resolve().relative_to(base.prior.ROOT)): sha(x) for x in paths},
        seed=base.SEEDS, steps=STEPS, batch=128, threads=4, workers=0, arms=ARMS, quotas=QUOTAS,
        original_2pct=True, external_outcomes_read=False, same_action_across_screens=True), immutable=True)
    return sha(p), source, forecasts, f, fixed, cap


def offsets(cv, ols, d, cut):
    names, index = np.unique(d['recording'], return_inverse=True)
    known = base.parent.paired(ols, cv, d['future'], d['valid'])[1].sum(1)
    ols0 = np.bincount(index, weights=d['weight']*known)
    cap = .02*np.bincount(index, weights=oracle.moments(cv, cv, d['future'], d['valid'], cut[0])['ref_easy'])
    return names, index, ols0, cap


def examples(d, donor, excluded, seed, bank, cut):
    reject = sorted([*excluded, donor])
    if len(set(reject)) != 3:
        raise ValueError('Overlapping donor')
    tag = base.triple_id(reject, seed)
    net = base.load_net(base.LOCAL/f'{tag}.pt', bank['triple_checkpoints'][tag],
                        old.motion.motion.network(17), reject)
    t = action.labels(d['file'])
    m = action.study.menu(dict(history=d['history'], query_counts=d['counts'], recording=t['recording'],
        tensors=(torch.from_numpy(d['x'][:, :17]), torch.from_numpy(d['cv'])),
        rotation=d['rotation'], scale=d['scale']), net)
    _, ix, ols0, cap = offsets(m['cv'], m['ols4'], t, cut)
    rng = np.random.default_rng([seed, *(base.SITES.index(s) for s in reject), base.SITES.index(donor)])
    x, y = [], []
    for lo, hi in m['slots'].values():
        q = slice(lo, hi)
        cv, ols = m['cv'][q], m['ols4'][q]
        legal = m['peak'][q] > 0.
        rec = ix[lo]
        for _ in range(action.DRAWS):
            rows = rng.choice(np.flatnonzero(legal.any(1)), 2, replace=False)
            family = [rng.choice(np.flatnonzero(legal[r])) for r in rows]
            anchor = cv.copy()
            anchor[rows] = m['paths'][lo+rows, family]
            feats, _ = action.inputs(cv, anchor, rows, d, q)
            win = oracle.legal_wins(cv, ols, t['future'][q], t['valid'][q], t['weight'][q],
                                    anchor, rows, cut, ols0[rec], cap[rec])
            x.append(model.summarize(feats, hi-lo))
            y.append(float(win.any()))
    if not x:
        raise ValueError('No donor opportunity examples')
    return torch.from_numpy(np.stack(x)), torch.tensor(y, dtype=torch.float32)


def fit(identity, batches):
    saved = old.LOCAL
    try:
        old.LOCAL = LOCAL
        torch.manual_seed(identity['seed'])
        return old.fit(identity, batches, model.loss, model.head(), STEPS, 128)
    finally:
        old.LOCAL = saved


@torch.inference_mode()
def propose(packet, d, ego, opportunity, previous):
    cv, anchor = packet['cv'], pair.prediction(packet, 'additive')
    obs, summary = [], []
    for (lo, hi), rows in zip(packet['bounds'], packet['additive_rows'], strict=True):
        q = slice(lo, hi)
        feats, _ = action.inputs(cv[q], anchor[q], rows-lo, d, q)
        obs.append(feats)
        summary.append(model.summarize(feats, hi-lo))
    pred = ego(torch.from_numpy(np.stack(obs))).numpy()
    rows, score = zip(*(model.nominate(p) for p in pred), strict=True)
    pick, margin = np.asarray(rows), np.asarray(score)
    p = opportunity(torch.from_numpy(np.stack(summary)))[:, 0].sigmoid().numpy()
    old_mask = previous['choices'][action.ARMS.index('ego')]
    on = old_mask[:, 0] >= 0
    np.testing.assert_array_equal(pick[on], old_mask[on])
    np.testing.assert_array_equal(margin > 0., on)
    return dict(choices=pick, old_mask=on, scores=np.stack((margin, p, p*expit(10.*margin))))


def observed(packet, d, cut, choices):
    cv, y, v, w = packet['cv'], d['future'], d['valid'], d['weight']
    anchor, rotated = pair.prediction(packet, 'additive'), action.paths(packet, choices)
    reference = pair.evaluate(anchor, packet, d, cut)
    changed = pair.evaluate(rotated, packet, d, cut)
    plus = base.parent.paired(anchor, rotated, y, v)[1].sum(1)*w
    minus = base.parent.paired(rotated, anchor, y, v)[1].sum(1)*w
    qg = np.array([(plus[lo:hi].sum(), minus[lo:hi].sum()) for lo, hi in packet['bounds']])
    return anchor, rotated, reference, changed, qg


def freeze(key, data, upstream, fixed, bank):
    cells = {}
    for name, c in upstream['cells'].items():
        f = fixed['cells'][name]
        path = LOCAL/f'{name}.json'
        excluded, seed = sorted(c['packets']), c['seed']
        donors = c['fit_sites']
        if set(donors) != set(base.SITES)-set(excluded) or f['fit_sites'] != donors:
            raise ValueError('Source roles changed')
        if not path.exists():
            train = [examples(data[s], s, excluded, seed, bank, c['cut']) for s in donors]
            ident = dict(name=name, registration=key, excluded=excluded, fit_sites=donors, seed=seed)
            net, ref = fit(ident, train)
            ego_ref = f['checkpoints']['ego']
            ego = base.load_net(action.LOCAL/f"{ego_ref['identity']['name']}.pt", ego_ref, action.model.AngleValue(False), excluded)
            scores, rules = {}, {}
            for s, r in c['packets'].items():
                with packet_file(ctx.study.LOCAL, r) as old_packet, packet_file(action.LOCAL, f['packets'][s]) as prev:
                    result = propose(old_packet, data[s], ego, net, prev)
                    _, _, _, v, gain = observed(old_packet, action.labels(data[s]['file']), c['cut'], result['choices'])
                    passes = v['dual'] & (gain[:, 0] > 0.)
                    rules[s] = {a: base.calibration(z, passes) for a, z in zip(ARMS, result['scores'], strict=True)}
                p = LOCAL/f'{name}_{s}.npz'
                scores[s] = dict(file=p.name, sha256=base.parent.indices.save_npz(p, result), incumbent=r['sha256'])
            write(path, dict(registration=key, fit_sites=donors, examples=sum(len(y) for x, y in train),
                positives=int(sum(y.sum().item() for x, y in train)), checkpoint=ref, packets=scores,
                thresholds_from_calibration_site=rules), immutable=True)
        row = read(path)
        if (row['registration'] != key or sha(LOCAL/f'{name}.pt') != row['checkpoint']['sha256']
                or any(sha(LOCAL/r['file']) != r['sha256'] for r in row['packets'].values())):
            raise ValueError('Changed opportunity checkpoint or frozen scores')
        cells[name] = row
        beat('freeze', len(cells), 63)
        if len(cells) % 7 == 0:
            print('Frozen:', len(cells), '/ 63', flush=True)
    write(OUT/'freeze.json', dict(registration=key, cells=cells, before_held_scoring=True), immutable=True)


def beat(phase, done, total):
    write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, completed=done, total=total))


def rank_metrics(score, y):
    pos = int(y.sum())
    neg = len(y)-pos
    if not pos or not neg:
        return None
    auc = (rankdata(score)[y].sum()-.5*pos*(pos+1))/(pos*neg)
    idx = np.argsort(-score, kind='stable')
    end = np.r_[np.flatnonzero(np.diff(score[idx]) != 0.), len(idx)-1]
    tp = y[idx].cumsum()[end]
    ap = ((np.diff(np.r_[0, tp])/pos)*(tp/(end+1))).sum()
    return dict(auc=float(auc), ap=float(ap))


def evaluate(packet, d, cut, frozen, thresholds):
    choices, old_mask, scores = (frozen[k] for k in ('choices', 'old_mask', 'scores'))
    cv, ols = packet['cv'], packet['ols4']
    anchor, changed, orig, edit, g = observed(packet, d, cut, choices)
    names, rec, ols0, cap = offsets(cv, ols, d, cut)
    y, v, w = (d[k] for k in ('future', 'valid', 'weight'))
    flag, chosen, relative = [], [], []
    rel = ctx.prior.prior.pair_reduction
    for (lo, hi), p, rot, name in zip(packet['bounds'], packet['additive_rows'], choices, packet['records'], strict=True):
        q = slice(lo, hi)
        i = rec[lo]
        if names[i] != name or not np.all(rec[q] == i):
            raise ValueError('Wrong recording bounds')
        win = oracle.legal_wins(cv[q], ols[q], y[q], v[q], w[q], anchor[q], p-lo, cut, ols0[i], cap[i])
        flag.append(bool(win.any()))
        chosen.append(bool(win[rot[0], rot[1]-1]))
        relative.append([rel(changed[None, q], anchor[q], y[q], v[q])[0, 1],
                         rel(anchor[None, q], changed[q], y[q], v[q])[0, 1]])
    exists, succeeds, r = np.asarray(flag), np.asarray(chosen), np.asarray(relative)
    if np.any(succeeds & ~(edit['dual'] & (g[:, 0] >= -1e-9))):
        raise ValueError('Oracle nomination failed the unchanged reference checks')
    selected = np.isin(d['recording'], packet['records'])
    mass = w[selected] @ np.where(v[selected], np.linalg.norm(cv[selected]-y[selected], axis=-1), 0.).sum(1)
    n = len(exists)
    sizes = (int(old_mask.sum()), int(np.ceil(.15*n)))
    masks, report = {}, {}
    for a, s in zip(ARMS, scores, strict=True):
        lim = thresholds[a][0]
        masks[a] = (*[model.top_mask(s, k) for k in sizes], s >= lim if lim is not None else np.zeros(n, dtype=bool))
        report[a] = dict(existence=rank_metrics(s, exists), nominee=rank_metrics(s, succeeds))
        for q, take in zip(QUOTAS, masks[a], strict=True):
            report[a][q] = dict(count=int(take.sum()), available=int(exists[take].sum()), strict=int(succeeds[take].sum()),
                actual_risk=int(edit['risk'][take].sum()), actual_dual=int(edit['dual'][take].sum()),
                positive_incremental=int((edit['dual'] & (g[:, 0] > 0.))[take].sum()),
                dual_all=int(np.where(take, edit['dual'], orig['dual']).sum()),
                lower_pp=float(100.*g[take, 0].sum()/mass), relative=float(r[take, 0].sum()/n))
    if not np.array_equal(masks['margin'][0], old_mask):
        raise ValueError('Equal-count margin control does not reproduce the original ego decision')
    paired = {}
    for a in ARMS[1:]:
        for q, x, b in zip(QUOTAS, masks[a], masks['margin'], strict=True):
            left, right = x & ~b, b & ~x
            paired[f'{a}_{q}'] = dict(lower_pp=float(100.*(g[left, 0].sum()+g[right, 1].sum())/mass),
                relative=float((r[left, 0].sum()+r[right, 1].sum())/n),
                strict_rate_pp=float(100.*(succeeds[x].sum()-succeeds[b].sum())/n))
    return dict(views=n, exists=int(exists.sum()), nominee_strict=int(succeeds.sum()),
                original_dual=int(orig['dual'].sum()), arms=report, paired=paired)


def score(key, data, parent, cap):
    path = OUT/'freeze.json'
    frozen = read(path)
    fr = sha(path)
    if frozen['registration'] != key or not frozen['before_held_scoring'] or set(frozen['cells']) != set(parent['cells']):
        raise ValueError('Require complete pre-score freeze')
    rows = []
    for name, c in parent['cells'].items():
        f = frozen['cells'][name]
        for s, r in c['packets'].items():
            cal, = set(c['packets'])-{s}
            z = f['packets'][s]
            if s in f['fit_sites'] or cal in f['fit_sites'] or z['incumbent'] != r['sha256']:
                raise ValueError('Changed fit or calibration roles')
            file = LOCAL/f'{name}_{s}_score.json'
            ident = dict(freeze=fr, site=s, seed=c['seed'], calibration=cal, packet=z['sha256'])
            if not file.exists():
                with packet_file(ctx.study.LOCAL, r) as p, packet_file(LOCAL, z) as proposals:
                    out = evaluate(p, action.labels(data[s]['file']), c['cut'], proposals, f['thresholds_from_calibration_site'][cal])
                write(file, dict(identity=ident, **out), immutable=True)
            out = read(file)
            if out['identity'] != ident:
                raise ValueError('Unbound held evaluation')
            rows.append(out)
        beat('score', len(rows), 126)
    totals = {k: sum(r[k] for r in rows) for k in ('views', 'exists', 'nominee_strict', 'original_dual')}
    if len(rows) != 126 or (totals['views'], totals['exists'], totals['original_dual']) != (8874, cap['strict_one_rotation_capacity'], 2975):
        raise ValueError('Changed exact oracle or original control')
    sites = [[r for r in rows if r['identity']['site'] == s] for s in base.SITES]
    draws = np.random.default_rng(2917).integers(7, size=(3000, 7))

    def ci(fn):
        def avg(seq):
            v = [z for z in seq if z is not None]
            return np.mean(v) if v else None
        v = np.array([avg(avg(fn(r) for r in g if r['identity']['seed'] == seed) for seed in base.SEEDS)
                      for g in sites], dtype=float)
        if not np.isfinite(v).all():
            return dict(estimable=False, locations=int(np.isfinite(v).sum()))
        return dict(point=float(v.mean()), ci95=np.quantile(v[draws].mean(1), [.025, .975]).tolist(),
                    by_site=dict(zip(base.SITES, v.tolist(), strict=True)))

    counts = {a: {q: dict(sum((Counter({k: z for k, z in r['arms'][a][q].items() if isinstance(z, int)})
                       for r in rows), Counter())) for q in QUOTAS} for a in ARMS}
    ranking = {a: {y: {m: ci(lambda r: (r['arms'][a][y] or {}).get(m)) for m in ('auc', 'ap')}
                  for y in ('existence', 'nominee')} for a in ARMS}
    paired = {a: {k: ci(lambda r: r['paired'][a][k]) for k in z} for a, z in rows[0]['paired'].items()}
    result = dict(result_source='fresh_run', registration=key, freeze=fr, totals=totals, counts=counts,
        ranking=ranking, paired=paired, training_examples=sum(r['examples'] for r in frozen['cells'].values()),
        training_positives=sum(r['positives'] for r in frozen['cells'].values()), fits=63, updates=63*STEPS, recordings=493,
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
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, source, bank, parent, fixed, cap = register()
        data = ctx.load_features(source)
        if phase == 'freeze':
            freeze(key, data, parent, fixed, bank)
        else:
            score(key, data, parent, cap)
