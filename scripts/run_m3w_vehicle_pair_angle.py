"""Test direct versus anchor-difference supervision on identical angle pairs."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_angle_agreement as old
from src.world_model import m3w_vehicle_pair_angle as model

short = old.old
action, ctx, nested, prior = short.action, short.ctx, short.nested, short.prior
np, torch, screen, base, read, write, sha = old.np, old.torch, old.screen, old.base, old.read, old.write, old.sha
OUT = old.OUT.parent/'opendd_vehicle_pair_angle_v1'
LOCAL = old.LOCAL.parent/OUT.name
MODES = ('anchor_difference', 'direct')
STEPS = 200
TOL = 1e-10


def register():
    key, source, bank, parent, teachers, chooser, _ = short.register()
    path = short.OUT/'freeze.json'
    first = read(path)
    a, b = (old.OUT/k for k in ('freeze.json', 'readout.json'))
    f, r = read(a), read(b)
    if (first['registration'] != key or not first['before_held_scoring'] or r['freeze'] != sha(a)
            or not r['original_2pct'] or r['external_outcomes_read'] or r['independent_confirmation']
            or not f['before_held_scoring'] or r['directions'] != 126):
        raise ValueError('Need verified fixed-angle comparison')
    paths = (Path(__file__), Path(model.__file__), Path(old.__file__), Path(short.__file__), path, a, b, OUT/'protocol.md')
    p = OUT/'registration.json'
    write(p, dict(files={str(x.resolve().relative_to(base.prior.ROOT)): sha(x) for x in paths},
                  seeds=base.SEEDS, steps=STEPS, batch=256, threads=4, workers=0, models=MODES,
                  original_2pct=True, external_outcomes_read=False), immutable=True)
    return sha(p), ctx.load_features(source), bank, parent, teachers, chooser, first, f, r


def scale(paths):
    diff = paths[1:]-paths[1]
    i, j = np.triu_indices(diff.shape[1], 1)
    return np.column_stack((np.linalg.norm(diff, axis=-1).sum((1, 2))/diff.shape[2],
                            np.linalg.norm(diff[:, i]-diff[:, j], axis=-1).mean((1, 2))))


def targets(paths, y, valid, bound, dual):
    lower = base.parent.paired
    traj = lambda a, b: lower(a, b, y, valid)[1].sum((1, 2))/paths.shape[2]
    direct = np.column_stack((traj(paths[1], paths[2:]), short.PAIR(paths[2:], paths[1], y, valid)[:, 1]))
    anchor = np.column_stack((traj(paths[0], paths[1:]), short.PAIR(paths[1:], paths[0], y, valid)[:, 1]))
    proxy = anchor[1:]-anchor[0]
    limit = bound[1:]+2e-5
    if np.any(np.abs(direct) > limit) or np.any(np.abs(proxy) > limit):
        raise ValueError('Angle pair target exceeds its geometry envelope')
    unit = bound[1:].clip(1e-12)
    return np.column_stack(((proxy/unit).clip(-1, 1), (direct/unit).clip(-1, 1), dual[2:])).astype(np.float32)


@torch.inference_mode()
def donor(d, s, excluded, seed, bank, teachers, chooser, cut):
    ex = sorted([*excluded, s])
    if len(set(ex)) != 3:
        raise ValueError('Donor and outer sites must be excluded from upstream fits')
    tag = nested.tag(ex, seed)
    e = base.load_net(prior.prior.LOCAL/f'{tag}_ego.pt', teachers['checkpoints']['ego'][tag],
                      action.model.AngleValue(False), ex)
    a = base.load_net(prior.LOCAL/f'{tag}_additive.pt', chooser['fits'][tag]['checkpoint'],
                      action.study.model.PairValue(False), ex)
    t = base.triple_id(ex, seed)
    forecaster = base.load_net(base.LOCAL/f'{t}.pt', bank['triple_checkpoints'][t],
                               nested.old.motion.motion.network(17), ex)
    d_y = action.labels(d['file'])
    m = action.study.menu(nested.data_slice(d, d_y), forecaster)
    _, rec, off, cap = screen.offsets(m['cv'], m['ols4'], d_y, cut)
    x, labels = [], []
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
        feat, _ = action.inputs(cv, anchor, sel, d, q)
        indices, margin = short.model.order(e(torch.from_numpy(feat)).numpy())
        paths = short.options(cv, anchor, sel, indices)
        y, v, w = (d_y[k][q] for k in ('future', 'valid', 'weight'))
        if not np.all(w == w[0]):
            raise ValueError('Training comparison requires equal within-recording weights')
        _, dual, _, _, _ = short.outcomes(cv, ols, anchor, paths, sel, y, v, w, cut, off[rec[lo]], cap[rec[lo]])
        bound = scale(paths)
        x.append(model.features(short.feature_rows(feat, indices, margin, hi-lo), bound)[0])
        labels.append(targets(paths, y, v, bound, dual))
    return tuple(torch.from_numpy(np.concatenate(a)) for a in (x, labels))


def fit(identity, data, direct):
    saved = nested.old.LOCAL
    try:
        nested.old.LOCAL = LOCAL
        torch.manual_seed(identity['seed'])
        return nested.old.fit(identity, data, lambda m, b: model.loss(m, b, direct), model.network(), STEPS, 256)
    finally:
        nested.old.LOCAL = saved


@torch.inference_mode()
def propose(p, z, d, ego, nets):
    cv = p['cv']
    anchor = screen.pair.prediction(p, 'additive')
    samples = []
    for (lo, hi), rows, indices in zip(p['bounds'], p['additive_rows'], z['shortlist'], strict=True):
        q, sel = slice(lo, hi), rows-lo
        feat, _ = action.inputs(cv[q], anchor[q], sel, d, q)
        actual, margin = short.model.order(ego(torch.from_numpy(feat)).numpy())
        np.testing.assert_array_equal(actual, indices)
        paths = short.options(cv[q], anchor[q], sel, indices)
        samples.append(model.features(short.feature_rows(feat, indices, margin, hi-lo), scale(paths)))
    x, ratio = (np.stack(a) for a in zip(*samples, strict=True))
    result = {}
    for tag, net in zip(MODES, nets, strict=True):
        t, a, score = model.choose(net(torch.from_numpy(x)).numpy(), ratio)
        result.update({f'{tag}_{k}': v for k, v in zip(('trajectory', 'agreement', 'score'), (t, a, score), strict=True)})
    return result


def heartbeat(phase, n):
    write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, completed=n, total=63))
    if n % 7 == 0:
        print(phase, n, '/ 63', flush=True)


def pair_contrasts(p, label, cut, z, packets, rules):
    arms = old.ARMS[2:]
    selected = {}
    masks = old.masks(z)
    for q, at in masks.items():
        for tag, a in packets.items():
            t = rules[tag][q][0]
            keep = a['score'] >= t if t is not None else np.zeros_like(at)
            selected[q, tag] = np.stack((a['trajectory'], a['agreement'], np.where(keep, a['agreement'], 0)))
    values = {tag: {q: np.zeros((len(arms), len(old.FIELDS))) for q in masks}
              for tag in ('pointwise', 'anchor_difference')}
    for i, (paths, y, valid, w, _) in enumerate(short.truth(p, label, cut, z['shortlist'])):
        for q, at in masks.items():
            if not at[i]:
                continue
            a = selected[q, 'direct'][:, i]+1
            for tag in values:
                b = selected[q, tag][:, i]+1
                for k in np.flatnonzero(a != b):
                    left, right = paths[a[k]], paths[b[k]]
                    gain, lower = base.parent.paired(right, left, y, valid)
                    rel = short.PAIR(left[None], right, y, valid)[0, 1]
                    values[tag][q][k] += w@lower.sum(1), rel, w@gain[:, -1]
    at = np.isin(label['recording'], p['records'])
    w = label['weight'][at]
    err = np.where(label['valid'][at], np.linalg.norm(p['cv'][at]-label['future'][at], axis=-1), 0.)
    denom = np.array([w@err.sum(1)/100., len(z['shortlist']), w@err[:, -1]/100.])
    if not (denom > 0.).all():
        raise ValueError('Missing paired reference denominators')
    return {tag: {q: {a: dict(zip(old.FIELDS, line, strict=True)) for a, line in zip(arms, gain/denom, strict=True)}
                  for q, gain in v.items()} for tag, v in values.items()}


def freeze(key, data, bank, parent, teachers, chooser, ref):
    old_action = read(action.OUT/'freeze.json')
    cells = {}
    for name, c in parent['cells'].items():
        src = ref['cells'][name]
        seed, excluded = c['seed'], sorted(c['packets'])
        donors = src['fit_sites']
        if set(donors) != set(base.SITES)-set(excluded):
            raise ValueError('Changed paired fit roles')
        p = LOCAL/f'{name}.json'
        if not p.exists():
            train = [donor(data[s], s, excluded, seed, bank, teachers, chooser, c['cut']) for s in donors]
            y = torch.cat([b[1] for b in train])
            if 3*len(y) != 2*src['training']['rows']:
                raise ValueError('Pair rows do not match the previous donor roster')
            stats = dict(rows=len(y), trajectory_sign_changed=int(((y[:, 0] > TOL) != (y[:, 2] > TOL)).sum()),
                         relative_sign_changed=int(((y[:, 1] > TOL) != (y[:, 3] > TOL)).sum()),
                         proxy_joint=int((y[:, :2] > TOL).all(1).sum()),
                         direct_joint=int((y[:, 2:4] > TOL).all(1).sum()))
            fits = [fit(dict(name=f'{name}_{tag}', registration=key, excluded=excluded, fit_sites=donors, seed=seed),
                        train, tag == 'direct') for tag in MODES]
            e = old_action['cells'][name]['checkpoints']['ego']
            ego = base.load_net(action.LOCAL/f"{e['identity']['name']}.pt", e, action.model.AngleValue(False), excluded)
            packets, cutoffs = {}, {}
            for s, packet in c['packets'].items():
                zref = src['packets'][s]
                with screen.packet_file(ctx.study.LOCAL, packet) as p0, screen.packet_file(short.LOCAL, zref) as z:
                    a = propose(p0, z, data[s], ego, [net for net, _ in fits])
                    m = old.masks(z)
                    flag, gain = old.compare(p0, action.labels(data[s]['file']), c['cut'], z, m['ego_count'] | m['fixed_15pct'])
                    cutoffs[s] = {}
                    for tag in MODES:
                        choice = a[f'{tag}_agreement']
                        ix = (np.arange(len(choice)), choice)
                        good = flag[ix][:, 1] & (gain[ix][:, :2] > TOL).all(1)
                        cutoffs[s][tag] = {q: old.model.threshold(a[f'{tag}_score'][at], good[at]) for q, at in m.items()}
                dest = LOCAL/f'{name}_{s}.npz'
                packets[s] = dict(file=dest.name, sha256=base.parent.indices.save_npz(dest, a), shortlist=zref['sha256'])
            write(p, dict(registration=key, donors=donors, training=stats, checkpoints=[r for _, r in fits],
                          packets=packets, cutoffs=cutoffs), immutable=True)
        r = read(p)
        if (r['registration'] != key or r['donors'] != donors
                or any(sha(LOCAL/f"{x['identity']['name']}.pt") != x['sha256'] for x in r['checkpoints'])
                or any(sha(LOCAL/x['file']) != x['sha256'] for x in r['packets'].values())):
            raise ValueError('Changed pair freeze or checkpoint')
        cells[name] = r
        heartbeat('freeze', len(cells))
    write(OUT/'freeze.json', dict(registration=key, before_held_scoring=True, cells=cells), immutable=True)


def score(key, data, parent, ref, prior_frozen, baseline):
    fp = OUT/'freeze.json'
    f = read(fp)
    fr = sha(fp)
    if f['registration'] != key or not f['before_held_scoring'] or set(f['cells']) != set(parent['cells']):
        raise ValueError('Require complete pair freeze')
    rows = []
    for name, c in parent['cells'].items():
        r = f['cells'][name]
        for s, packet in c['packets'].items():
            cal, = set(c['packets'])-{s}
            zref, pref = ref['cells'][name]['packets'][s], r['packets'][s]
            if pref['shortlist'] != zref['sha256'] or set(r['donors']) & {cal, s}:
                raise ValueError('Changed shortlist or source roles')
            prior = prior_frozen['directions'][f'{name}_{s}']
            path = LOCAL/f'{name}_{s}_score.json'
            identity = dict(freeze=fr, site=s, seed=c['seed'], calibration=cal, packet=pref['sha256'])
            if not path.exists():
                with (screen.packet_file(ctx.study.LOCAL, packet) as p,
                      screen.packet_file(short.LOCAL, zref) as z, screen.packet_file(LOCAL, pref) as a,
                      screen.packet_file(old.LOCAL, prior) as u):
                    label = action.labels(data[s]['file'])
                    packets = dict(pointwise=u, **{tag: dict(utility=u['utility'], **{
                        k: a[f'{tag}_{k}'] for k in ('trajectory', 'agreement', 'score')}) for tag in MODES})
                    rules = dict(pointwise=prior_frozen['directions'][f'{name}_{cal}']['cutoffs'], **r['cutoffs'][cal])
                    results = {tag: old.metrics(p, label, c['cut'], z, values, rules[tag]) for tag, values in packets.items()}
                    paired = pair_contrasts(p, label, c['cut'], z, packets, rules)
                write(path, dict(identity=identity, results=results, paired=paired), immutable=True)
            v = read(path)
            if v['identity'] != identity:
                raise ValueError('Changed pair score direction')
            rows.append(v)
        heartbeat('score', len(rows)//2)
    groups = [[r for r in rows if r['identity']['site'] == s] for s in base.SITES]
    draws = np.random.default_rng(2917).integers(7, size=(3000, 7))

    def ci(fn):
        means = np.asarray([np.mean([np.mean([fn(r) for r in g if r['identity']['seed'] == seed])
                                    for seed in base.SEEDS]) for g in groups])
        return dict(point=float(means.mean()), ci95=np.quantile(means[draws].mean(1), [.025, .975]).tolist(),
                    by_site=dict(zip(base.SITES, means.tolist(), strict=True)))

    counts, effects, contrasts = {}, {}, {}
    for tag in ('pointwise', *MODES):
        arms = old.ARMS if tag == 'pointwise' else old.ARMS[2:]
        counts[tag] = {q: {a: dict(sum((Counter(r['results'][tag]['counts'][q][a]) for r in rows), Counter()))
                           for a in arms} for q in old.QUOTAS}
        effects[tag] = {q: {a: {k: ci(lambda r: r['results'][tag]['effects'][q][a][k]) for k in old.FIELDS}
                            for a in arms} for q in old.QUOTAS}
    if counts['pointwise'] != baseline['counts'] or effects['pointwise'] != baseline['effects']:
        raise ValueError('Previous fixed-angle comparison must reproduce')
    for against in ('pointwise', 'anchor_difference'):
        contrasts[against] = {q: {a: {k: ci(lambda r: r['paired'][against][q][a][k]) for k in old.FIELDS}
                                  for a in old.ARMS[2:]} for q in old.QUOTAS}
    stats = dict(sum((Counter(c['training']) for c in f['cells'].values()), Counter()))
    calibrated = {tag: {q: sum(t[tag][q][0] is not None for c in f['cells'].values() for t in c['cutoffs'].values())
                        for q in old.QUOTAS} for tag in MODES}
    result = dict(result_source='fresh_run', freeze=fr, fits=126, updates=126*STEPS, directions=len(rows), training=stats,
                  counts=counts, effects=effects, direct_minus=contrasts, calibration_nonempty=calibrated,
                  capacity=baseline['capacity'], original_2pct=True, external_outcomes_read=False,
                  independent_confirmation=False, deployment_changed=False)
    write(OUT/'readout.json', result, immutable=True)
    heartbeat('complete', len(rows)//2)
    print(stats, counts, calibrated, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('freeze', 'score'))
    phase = cli.parse_args().phase
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use the native arm64 .venv-pytorch interpreter')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, bank, parent, teachers, chooser, prior_short, old_freeze, old_readout = register()
        if phase == 'freeze':
            freeze(key, data, bank, parent, teachers, chooser, prior_short)
        else:
            score(key, data, parent, prior_short, old_freeze, old_readout)
