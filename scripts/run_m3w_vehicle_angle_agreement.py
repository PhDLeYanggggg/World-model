"""Freeze and score angle-only changes to the OpenDD shortlist policies."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_shortlist as old
from src.world_model import m3w_vehicle_angle_agreement as model

np, torch, screen, base = old.np, old.torch, old.screen, old.base
read, write, sha = old.read, old.write, old.sha
OUT = old.OUT.parent/'opendd_vehicle_angle_agreement_v1'
LOCAL = old.LOCAL.parent/OUT.name
QUOTAS = ('ego_count', 'fixed_15pct')
ARMS = ('margin', 'utility', 'trajectory', 'agreement', 'calibrated_agreement')
FIELDS = ('lower_pp', 'relative', 'observed_fde_pp')


def register():
    key, source, _, parent, _, _, _ = old.register()
    freeze, result = (old.OUT/k for k in ('freeze.json', 'readout.json'))
    first = read(freeze)
    ref = read(result)
    if (first['registration'] != key or ref['freeze'] != sha(freeze)
            or not first['before_held_scoring'] or ref['external_outcomes_read']):
        raise ValueError('Require verified previous shortlist and readout')
    paths = (Path(__file__), Path(model.__file__), Path(old.__file__), Path(old.model.__file__),
             freeze, result, OUT/'protocol.md')
    p = OUT/'registration.json'
    write(p, dict(files={str(x.resolve().relative_to(base.prior.ROOT)): sha(x) for x in paths},
                  seeds=base.SEEDS, quotas=QUOTAS, arms=ARMS, fits=0, original_2pct=True,
                  external_outcomes_read=False, same_intervention_masks=True), immutable=True)
    return sha(p), old.ctx.load_features(source), parent, first, ref


@torch.inference_mode()
def propose(p, z, d, net):
    cv = p['cv']
    anchor = screen.pair.prediction(p, 'additive')
    features, envelopes, counts = [], [], []
    for (lo, hi), rows, short in zip(p['bounds'], p['additive_rows'], z['shortlist'], strict=True):
        q, sel = slice(lo, hi), rows-lo
        f, _ = old.action.inputs(cv[q], anchor[q], sel, d, q)
        a, margin = old.model.order(net['ego'](torch.from_numpy(f)).numpy())
        np.testing.assert_array_equal(a, short)
        features.append(old.feature_rows(f, short, margin, hi-lo))
        paths = old.options(cv[q], anchor[q], sel, short)
        envelopes.append(np.linalg.norm(paths[1:]-paths[0], axis=-1).sum((1, 2))/paths.shape[2])
        counts.append(hi-lo)
    logits = net['utility'](torch.from_numpy(np.stack(features))).numpy()
    envelope = np.asarray(envelopes)
    _, old_scores = old.model.scores(logits, envelope)
    utility = old_scores.argmax(1)
    np.testing.assert_array_equal(utility, z['choices'][old.ARMS.index('utility_fixed')])
    t, both, score = model.choices(logits, envelope, np.asarray(counts))
    return dict(utility=utility, trajectory=t, agreement=both, score=score)


def masks(z):
    n = len(z['old_mask'])
    return dict(ego_count=z['old_mask'],
                fixed_15pct=screen.model.top_mask(z['scores'][0], int(np.ceil(.15*n))))


def compare(p, d, cut, z, keep):
    """Direct bounds versus the first angle, never subtract two bounds."""
    n = len(z['shortlist'])
    flags = np.zeros((n, 3, 4), dtype=bool)
    gain = np.zeros((n, 3, 3))
    for k, (paths, y, valid, w, out) in enumerate(old.truth(p, d, cut, z['shortlist'])):
        if not keep[k]:
            continue
        safe, dual, inc, strict, _ = out
        flags[k] = np.column_stack((safe, dual, inc, strict))[1:]
        relative = old.PAIR(paths[1:], paths[1], y, valid)[:, 1]
        lower = (base.parent.paired(paths[1], paths[1:], y, valid)[1].sum(-1)) @ w
        fde = np.where(valid[:, -1], np.linalg.norm(paths[1:, :, -1]-y[:, -1], axis=-1), 0.) @ w
        gain[k] = np.column_stack((lower, relative, fde[0]-fde))
    if np.any(gain[:, 0]) or np.any(flags[~keep]):
        raise ValueError('Incumbent comparison or masks changed')
    return flags, gain


def freeze(key, data, parent, ref):
    old_action = read(old.action.OUT/'freeze.json')
    rows = {}
    for name, c in parent['cells'].items():
        refs = ref['cells'][name]['checkpoints']
        u, = [r for r in refs if r['identity']['name'].endswith('_utility')]
        e = old_action['cells'][name]['checkpoints']['ego']
        for s, packet in c['packets'].items():
            dest = LOCAL/f'{name}_{s}.json'
            zref = ref['cells'][name]['packets'][s]
            ident = dict(registration=key, direction=name, site=s, previous_packet=zref['sha256'])
            if not dest.exists():
                net = {
                    'ego': base.load_net(old.action.LOCAL/f"{e['identity']['name']}.pt", e,
                                         old.action.model.AngleValue(False), c['packets']),
                    'utility': base.load_net(old.LOCAL/f"{u['identity']['name']}.pt", u,
                                             old.model.network(), c['packets']),
                }
                with screen.packet_file(old.ctx.study.LOCAL, packet) as p, screen.packet_file(old.LOCAL, zref) as z:
                    a = propose(p, z, data[s], net)
                    m = masks(z)
                    d = old.action.labels(data[s]['file'])
                    flag, gain = compare(p, d, c['cut'], z, m['ego_count'] | m['fixed_15pct'])
                    r = np.arange(len(a['agreement']))
                    ix = (r, a['agreement'])
                    good = flag[ix][:, 1] & (gain[ix][:, :2] > 1e-10).all(1)
                    cutoffs = {q: model.threshold(a['score'][at], good[at]) for q, at in m.items()}
                path = LOCAL/f'{name}_{s}.npz'
                entry = dict(identity=ident, file=path.name, sha256=base.parent.indices.save_npz(path, a), cutoffs=cutoffs)
                write(dest, entry, immutable=True)
            entry = read(dest)
            if entry['identity'] != ident or sha(LOCAL/entry['file']) != entry['sha256']:
                raise ValueError('Changed scores or calibration')
            rows[f'{name}_{s}'] = entry
        heartbeat('freeze', len(rows))
    write(OUT/'freeze.json', dict(registration=key, before_held_scoring=True, directions=rows), immutable=True)


def heartbeat(phase, n):
    write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, completed=n, total=126))
    if n % 14 == 0 or n == 126:
        print(phase, n, '/ 126', flush=True)


def metrics(p, d, cut, z, a, rules):
    m = masks(z)
    flags, gain = compare(p, d, cut, z, m['ego_count'] | m['fixed_15pct'])
    at = np.isin(d['recording'], p['records'])
    err = np.where(d['valid'][at], np.linalg.norm(p['cv'][at]-d['future'][at], axis=-1), 0.)
    w = d['weight'][at]
    denom = np.array([w@err.sum(1)/100., len(z['shortlist']), w@err[:, -1]/100.])
    if not (denom > 0.).all():
        raise ValueError('Need CV denominators for both paired effects')
    counts, effects, capacity = {}, {}, {}
    for q, at in m.items():
        t = rules[q][0]
        guard = a['agreement']
        selected = dict(margin=np.zeros(len(guard), dtype=int), **{k: a[k] for k in ARMS[1:-1]},
                        calibrated_agreement=np.where(a['score'] >= t, guard, 0) if t is not None else np.zeros_like(guard))
        traj, pair = gain[at, :, 0] > 1e-10, gain[at, :, 1] > 1e-10
        joint = traj & pair
        capacity[q] = dict(trajectory=int(traj.any(1).sum()), relative=int(pair.any(1).sum()),
                           joint=int(joint.any(1).sum()), qualified=int((joint & flags[at, :, 1]).any(1).sum()))
        counts[q], effects[q] = {}, {}
        for arm, choice in selected.items():
            ix = (np.flatnonzero(at), choice[at])
            f, g = flags[ix], gain[ix]
            changed = choice[at] != 0
            both = (g[:, :2] > 1e-10).all(1)
            counts[q][arm] = dict(rotated=int(at.sum()), swaps=int(changed.sum()),
                risk=int(f[:, 0].sum()), dual=int(f[:, 1].sum()), incremental=int(f[:, 2].sum()), strict=int(f[:, 3].sum()),
                joint_gain=int(both.sum()), qualified_joint_gain=int((both & f[:, 1]).sum()),
                trajectory_harm=int((g[:, 0] < -1e-10).sum()), relative_harm=int((g[:, 1] < -1e-10).sum()))
            effects[q][arm] = dict(zip(FIELDS, (g.sum(0)/denom).tolist(), strict=True))
    return dict(counts=counts, effects=effects, capacity=capacity)


def score(key, data, parent, prior, reference):
    path = OUT/'freeze.json'
    frozen = read(path)
    digest = sha(path)
    if frozen['registration'] != key or not frozen['before_held_scoring'] or len(frozen['directions']) != 126:
        raise ValueError('Require complete pre-score freeze')
    rows = []
    for name, c in parent['cells'].items():
        for s, packet in c['packets'].items():
            cal, = set(c['packets'])-{s}
            pred = frozen['directions'][f'{name}_{s}']
            rules = frozen['directions'][f'{name}_{cal}']['cutoffs']
            ident = dict(freeze=digest, site=s, seed=c['seed'], calibration=cal, packet=pred['sha256'])
            dest = LOCAL/f'{name}_{s}_score.json'
            if not dest.exists():
                zref = prior['cells'][name]['packets'][s]
                with (screen.packet_file(old.ctx.study.LOCAL, packet) as p,
                      screen.packet_file(old.LOCAL, zref) as z, screen.packet_file(LOCAL, pred) as a):
                    r = metrics(p, old.action.labels(data[s]['file']), c['cut'], z, a, rules)
                write(dest, dict(identity=ident, **r), immutable=True)
            row = read(dest)
            if row['identity'] != ident:
                raise ValueError('Changed held direction')
            rows.append(row)
        heartbeat('score', len(rows))
    counts = {q: {a: dict(sum((Counter(r['counts'][q][a]) for r in rows), Counter()))
                  for a in ARMS} for q in QUOTAS}
    for q, table in counts.items():
        for a, b in (('margin', 'margin'), ('utility', 'utility_fixed')):
            expected = reference['counts'][b][q]
            if any(table[a].get(k, 0) != expected[k] for k in ('risk', 'dual', 'strict', 'incremental')):
                raise ValueError('Did not replay frozen shortlist comparator')
            if any(v['rotated'] != expected['count'] for v in table.values()):
                raise ValueError('Lost matched intervention count')
    draws = np.random.default_rng(2917).integers(7, size=(3000, 7))
    groups = [[r for r in rows if r['identity']['site'] == s] for s in base.SITES]

    def effect(q, a, k):
        means = np.asarray([np.mean([np.mean([r['effects'][q][a][k] for r in g if r['identity']['seed'] == seed])
                                    for seed in base.SEEDS]) for g in groups])
        return dict(point=float(means.mean()), ci95=np.quantile(means[draws].mean(1), [.025, .975]).tolist(),
                    by_site=dict(zip(base.SITES, means.tolist(), strict=True)))

    cap = {q: dict(sum((Counter(r['capacity'][q]) for r in rows), Counter())) for q in QUOTAS}
    effects = {q: {a: {k: effect(q, a, k) for k in FIELDS} for a in ARMS} for q in QUOTAS}
    for q in QUOTAS:
        for k in FIELDS:
            a = effects[q]['utility'][k]['by_site']
            b = reference['paired'][f'utility_fixed_minus_margin_{q}'][k]['by_site']
            np.testing.assert_allclose([a[s] for s in base.SITES], [b[s] for s in base.SITES], atol=1e-12, rtol=0.)
    calibrated = {q: sum(r['cutoffs'][q][0] is not None for r in frozen['directions'].values()) for q in QUOTAS}
    result = dict(result_source='fresh_run_frozen_model_replay', freeze=digest, directions=len(rows), fits=0,
                  counts=counts, capacity=cap, effects=effects, calibration_nonempty_directions=calibrated,
                  original_2pct=True, external_outcomes_read=False, independent_confirmation=False, deployment_changed=False)
    write(OUT/'readout.json', result, immutable=True)
    heartbeat('complete', len(rows))
    print(counts, cap, calibrated, flush=True)


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
        key, data, parent, prior, reference = register()
        if phase == 'freeze':
            freeze(key, data, parent, prior)
        else:
            score(key, data, parent, prior, reference)
