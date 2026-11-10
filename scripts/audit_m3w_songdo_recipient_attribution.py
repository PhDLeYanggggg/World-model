"""Explain frozen Songdo confirmation outcomes without fitting or threshold changes."""

from collections import Counter
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_m3w_songdo_confirmation as published
from scripts import run_m3w_songdo_confirmation as primary
from src.evaluation import m3w_recipient_effect_split as split

np, storage = primary.np, primary.pack
read = primary.first.read
OUT = published.PUBLIC.parent/'songdo_recipient_attribution_v1'
LOCAL = published.RUN.parent/OUT.name
CONTROLS = primary.protocol.ARMS[:-1]


def check_scope():
    old = published.secondary.read_finished(published.RUN)
    r = read(published.PUBLIC/'confirmation_receipt.json')
    reg = read(published.RUN/'registration.json')
    b = old['binding']
    root = sha256(''.join(sorted(b['flight_score_sha256'].values())).encode()).hexdigest()
    names = {n: g for n, (role, g) in reg['flight_roles'].items() if role == 'confirmation'}
    if (len(names) != 160 or len(set(names.values())) != 14 or old['matched_only']
            or old['matched_seed_views'] != 480 or root != r['binding']['flight_score_hash_root']
            or set(b['flight_score_sha256']) != set(names) or not r['original_2pct_unchanged']
            or any(b[k] != r['binding'][k] for k in ('registration_sha256', 'release_freeze_sha256', 'primary_score_sha256'))):
        raise ValueError('Require the exact already completed 160-flight confirmation')
    source = primary.Songdo()
    if reg['binding'] != source.binding:
        raise ValueError('Frozen source or scoring code changed')
    bind = dict(registration_sha256=b['registration_sha256'], release_freeze_sha256=b['release_freeze_sha256'],
                primary_score_sha256=b['primary_score_sha256'], flight_score_hash_root=root,
                protocol_sha256=storage.sha(OUT/'protocol.md'), driver_sha256=storage.sha(Path(__file__)),
                decomposition_sha256=storage.sha(Path(split.__file__)))
    return source, names, b['flight_score_sha256'], bind, r


def verified_flight(path, previous):
    if storage.sha(path) != previous:
        raise ValueError('An original score receipt changed')
    return read(path)


def flight(data, name, old):
    meta, cache, prefix, bank = data.observed(name)
    proposal = primary.checked(primary.PACKETS/f'{Path(name).stem}.npz', data.proposals[name]['packet_sha256'])
    result = {}
    with (np.load(prefix, allow_pickle=False) as p, np.load(bank, allow_pickle=False) as b,
          np.load(proposal, allow_pickle=False) as z):
        cv, off = b['cv'], b['offsets']
        np.testing.assert_array_equal(p['offsets'], off)
        y, valid = data.targets(name, cache, p, meta)
        w = np.repeat(1./np.diff(off), np.diff(off))
        mass = float(w @ np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
        for seed in primary.protocol.SEEDS:
            key = f'seed{seed}_'
            lo, hi = z[key+'bounds']
            rows = {a: z[key+a+'_rows'] for a in primary.protocol.ARMS}
            arms = {a: (row-lo, z[key+a+'_paths']) for a, row in rows.items()}
            old_view = old['scores'][str(seed)]['matched']
            if not old_view['matched_eligible'] or old_view['coverage']['agents'] != len(cv):
                raise ValueError('Different frozen proposal support')
            _, pred = primary.protocol.readout.matched_paths(cv[lo:hi], np.array([0, hi-lo]),
                dict(bounds=(0, hi-lo), radius=float(z[key+'radius'].item()), arms=arms))
            result[str(seed)] = {}
            for arm in CONTROLS:
                part = split.decompose(pred['joint'], pred[arm], y[lo:hi], valid[lo:hi],
                                       rows['joint']-lo, rows[arm]-lo, mass)
                total = {k: sum(x[k] for x in part.values()) for k in split.METRICS}
                old_ref = old_view['matched_contrasts'][arm]
                np.testing.assert_allclose([total['observed_pp'], total['lower_pp']],
                                           [old_ref['known_pp'], old_ref['lower_pp']], atol=1e-12, rtol=0.)
                allocation = {k: part['joint_only'][k]+part['control_only'][k] for k in split.METRICS}
                result[str(seed)][arm] = dict(**part, allocation=allocation, total=total)
    return result


def heartbeat(i, n, member):
    p = LOCAL/'heartbeat.pending'
    p.write_text(json.dumps(dict(pid=os.getpid(), completed=i, total=n, flight=member))+'\n')
    p.replace(LOCAL/'heartbeat.json')
    if i == 1 or i % 10 == 0:
        print(f'Attributed {i}/{n} previously scored flights', flush=True)


def aggregate(names, rows, original):
    by_source = {g: sorted(n for n, x in names.items() if x == g) for g in sorted(set(names.values()))}
    draws = np.random.default_rng(1439).integers(len(by_source), size=(3000, len(by_source)))
    summary, counts = {}, {}
    for a in CONTROLS:
        effects = {}
        count = Counter()
        for r in rows.values():
            for seed in r.values():
                view = seed[a]
                count[f'shared_recipients_{view["shared"]["rows"]}'] += 1
                for p in split.PARTS:
                    count[f'{p}_changed_rows'] += view[p]['different_paths']
        for p in (*split.PARTS, 'allocation', 'total'):
            effects[p] = {}
            for metric in split.METRICS:
                v = np.array([np.mean([np.mean([s[a][p][metric] for s in rows[n].values()]) for n in group])
                              for group in by_source.values()])
                effects[p][metric] = dict(point=float(v.mean()), source_means=v.tolist(),
                    ci95=np.quantile(v[draws].mean(1), [.025, .975]).tolist())
        for k, old in (('observed_pp', 'known_pp'), ('lower_pp', 'lower_pp')):
            item = effects['total'][k]
            ref = original['matched_joint_minus'][a][old]
            np.testing.assert_allclose(item['source_means'], ref['component_points'], rtol=0., atol=1e-12)
            np.testing.assert_allclose([item['point'], *item['ci95']], [ref['point'], *ref['ci95']], rtol=0., atol=1e-12)
        summary[a], counts[a] = effects, count
    return summary, counts


def run():
    data, names, old_scores, binding, original = check_scope()
    rows = {}
    for i, n in enumerate(sorted(names), 1):
        prev = verified_flight(published.RUN/'confirmation'/f'{Path(n).stem}.json', old_scores[n])
        p = LOCAL/f'{Path(n).stem}.json'
        key = dict(binding=binding, flight=n, primary_flight_score=old_scores[n])
        if not p.exists():
            published.secondary.write(p, dict(key, scores=flight(data, n, prev)))
        item = read(p)
        if any(item[k] != v for k, v in key.items()):
            raise ValueError('A decomposition receipt belongs to another experiment')
        rows[n] = item['scores']
        heartbeat(i, len(names), n)
    effects, counts = aggregate(names, rows, original)
    published.secondary.write(OUT/'readout.json', dict(result_source='fresh_retrospective_attribution',
        binding=binding, flights=len(rows), flight_seed_views=3*len(rows), source_components=len(set(names.values())),
        bootstrap_draws=3000, primary_scores_reproduced=True, controls=effects, choice_counts=counts,
        previously_scored_confirmation_labels_reread=True, new_cohort_opened=False, no_fit=True,
        calibration_outcomes_opened=False, original_2pct=True, release_decision_changed=False))
    print(dict(flights=len(rows), views=3*len(rows), components=len(set(names.values()))), flush=True)


if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
