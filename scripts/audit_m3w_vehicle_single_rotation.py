"""Test one-rotation repair against an identical-recipient control."""

from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_vehicle_angular_pair as prior
from src.world_model import m3w_vehicle_kinematic_blend as kinematic
from src.world_model import m3w_vehicle_single_rotation as model


study, base, pair, old, np, torch = prior.study, prior.base, prior.pair, prior.old, prior.np, prior.torch
sha, write = prior.sha, prior.write
OUT = prior.OUT.parent/'opendd_vehicle_single_rotation_v1'
LOCAL = prior.LOCAL.parent/OUT.name
ARMS = ('original', 'independent', 'interaction', 'independent_at_interaction')
CONTRASTS = tuple((k, 'original') for k in ARMS[1:])+(('interaction', 'independent_at_interaction'),)


def evaluate(packet, d, cut):
    cv, ols = packet['cv'], packet['ols4']
    h, y, valid, weight = (d[k] for k in ('history', 'future', 'valid', 'weight'))
    scale = np.linalg.norm(np.diff(h, axis=1), axis=-1).mean(1).clip(.05)
    b = kinematic.paths(h, cv, scale)
    surrogates = np.stack((ols, cv+b[..., 2]*scale[:, None, None], cv+b[..., 3]*scale[:, None, None]))
    anchor = pair.prediction(packet, 'additive')
    paths = {k: anchor.copy() for k in ARMS}
    count = Counter()
    rotates = dict.fromkeys(ARMS, 0)
    rel = {f'{a}_minus_{b}': np.zeros(2) for a, b in CONTRASTS}
    for k, (lo, hi) in enumerate(packet['bounds']):
        q = slice(lo, hi)
        rows = packet['additive_rows'][k]
        if rows.shape != (2,) or len(set(rows)) != 2 or not np.all((lo <= rows) & (rows < hi)):
            raise ValueError('Illegal recipient roster')
        bank = prior.model.candidates(cv[rows], anchor[rows])
        choices = model.choose(*model.proxy_gains(bank, anchor[q], surrogates[:, q], h[q], rows-lo, scale[q]))
        for arm, choice in choices.items():
            if choice is not None:
                row, angle = choice
                paths[arm][rows[row]] = bank[row, angle]
                rotates[arm] += 1
        if choices['interaction'] is not None and choices['interaction'][0] != choices['independent_at_interaction'][0]:
            raise ValueError('Matched comparison changed the rotating actor')
        count.update(views=1, known_pair_query=int((valid[q].sum(0) >= 2).any()))
        for a, b in CONTRASTS:
            rel[f'{a}_minus_{b}'] += prior.pair_reduction(paths[a][None, q], paths[b][q], y[q], valid[q])[0]
    result = {}
    for k, p in paths.items():
        x = pair.evaluate(p, packet, d, cut)
        result[k] = dict(dual=int(x['dual'].sum()), risk=int(x['risk'].sum()), missing=int(x['missing'].sum()),
                         rotated_views=rotates[k])
    if len({r['missing'] for r in result.values()}) != 1 or rotates['interaction'] != rotates['independent_at_interaction']:
        raise ValueError('Changed support or matched mask')
    roster = np.isin(d['recording'], packet['records'])
    mass = (weight[roster]*np.where(valid[roster], np.linalg.norm(cv[roster]-y[roster], axis=-1), 0.).sum(1)).sum()
    effects = {}
    for a, b in CONTRASTS:
        key = f'{a}_minus_{b}'
        g, lower = base.parent.paired(paths[b][roster], paths[a][roster], y[roster], valid[roster])
        effects[key] = dict(observed_pp=float(100.*(weight[roster]*g.sum(1)).sum()/mass),
            lower_pp=float(100.*(weight[roster]*lower.sum(1)).sum()/mass),
            pair_known=float(rel[key][0]/max(1, count['known_pair_query'])), pair_lower=float(rel[key][1]/count['views']))
    return dict(counts=dict(count), arms=result, contrasts=effects)


def summarize(rows):
    groups = {s: [r for r in rows if r['identity']['site'] == s] for s in base.SITES}
    draws = np.random.default_rng(2917).integers(len(groups), size=(3000, len(groups)))

    def ci(key, field):
        x = np.asarray([np.mean([np.mean([r['contrasts'][key][field] for r in group if r['identity']['seed'] == seed])
                                for seed in base.SEEDS]) for group in groups.values()])
        return dict(point=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist())

    return dict(counts=dict(sum((Counter(r['counts']) for r in rows), Counter())),
                arms={k: {f: sum(r['arms'][k][f] for r in rows) for f in ('dual','risk','missing','rotated_views')} for k in ARMS},
                paired={k: {f: ci(k, f) for f in rows[0]['contrasts'][k]} for k in rows[0]['contrasts']})


def main():
    source, _, frozen, earlier, _ = prior.ref.settings()
    p, r = (prior.OUT/k for k in ('registration.json', 'readout.json'))
    binding = old.read(r)
    if binding['registration'] != sha(p) or binding['external_outcomes_read'] or not binding['original_2pct']:
        raise ValueError('Require the completed two-rotation comparison')
    path = OUT/'registration.json'
    files = (Path(__file__), Path(model.__file__), Path(kinematic.__file__), OUT/'protocol.md', p, r)
    write(path, dict(inputs={str(q.resolve().relative_to(base.prior.ROOT)): sha(q) for q in files},
        original_2pct=True, external_outcomes_read=False, no_training=True, max_rotations=1), immutable=True)
    data, rows = base.parent.load(source), []
    key = sha(path)
    for name, cell in frozen['cells'].items():
        seed, excluded = cell['seed'], sorted(cell['packets'])
        if set(cell['fit_sites']) != set(base.SITES)-set(excluded):
            raise ValueError('Invalid additive training roles')
        for site, r in cell['packets'].items():
            f = LOCAL/f'{name}_{site}.json'
            ident = dict(registration=key, packet=r['sha256'], site=site, seed=seed)
            with prior.ref.prior.packet(study.LOCAL, r) as packet:
                if not f.exists():
                    write(f, dict(identity=ident, **evaluate(packet, data[site], cell['cut'])), immutable=True)
            row = old.read(f)
            if row['identity'] != ident:
                raise ValueError('Resumed record does not match the fixed roster')
            rows.append(row)
        write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), completed=len(rows), total=126))
        if len(rows) % 14 == 0:
            print(len(rows), '/ 126 complete', flush=True)
    result = summarize(rows)
    if (len(rows), result['counts']['views'], result['arms']['original']['dual']) != (
            126, 8874, earlier['summary']['additive']['matched_dual']):
        raise ValueError('Original additive control changed')
    write(OUT/'readout.json', dict(registration=key, summary=result, by_direction=rows,
        unique_recordings=binding['unique_recordings'], original_2pct=True, no_training=True,
        external_outcomes_read=False, independent_confirmation=False, deployment_changed=False), immutable=True)
    print(result, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
