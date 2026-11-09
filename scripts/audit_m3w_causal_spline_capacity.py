"""Add frozen learned trajectories to the exact-count capacity comparison."""
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_causal_velocity_spline as study
from scripts import audit_m3w_ols_anchor_menu as menu

old, base, prior, np = study.old, study.base, study.prior, study.np
FRACTIONS = (.15, .35, 1.)


def causal_bank(history):
    c = prior.study.motion.bank(history)
    cv = c[:, 0]
    ray = base.expanded(cv, c[:, 1:])
    anchor = np.stack([menu.previous.rollout(cv, ray, a)
                       for a in range(ray.shape[1]*menu.previous.amplitude.M)], axis=1)
    return np.concatenate((anchor, menu.model.candidates(history)), axis=1)


def main():
    cap, fr, rp = menu.OUT / 'readout.json', study.OUT / 'freeze.json', study.OUT / 'readout.json'
    capacity, freeze, readout = map(old.read, (cap, fr, rp))
    if (readout['freeze'] != old.sha(fr) or capacity['registration']['sha256'] != old.sha(menu.OUT/'registration.json')
            or not freeze['before_held_scoring'] or any(
                p['a_labels_opened'] or p['independent_confirmation'] or not p['original_2pct']
                for p in (capacity, readout))):
        raise ValueError('Require frozen development-only paths and counts')
    reg = study.OUT / 'capacity_registration.json'
    old.once(reg, old.stable(dict(parents=[old.artifact(f) for f in (cap, fr, rp)],
        code=old.artifact(Path(__file__)), fractions=FRACTIONS, modes=study.model.MODES,
        seconds_per_solve=menu.SECONDS, original_2pct=True, exact_query_counts=True, diagnostic_only=True)))
    target = {(x['key']['cell'], x['key']['recording']): x for x in capacity['views']}
    earlier = old.read(study.parent.OUT / 'freeze.json')
    out = study.PRIVATE / 'capacity'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.register()
        data = base.fit_population(lambda **_: None)
        for j, (c, initial) in enumerate(zip(freeze['cells'], earlier['cells'], strict=True)):
            i = c['identity']
            if i != initial['identity'] or c['cut'] != initial['cut']:
                raise ValueError('Frozen roles changed')
            _, h = base.subset(data, i['held'])
            with (np.load(old.checked(c['predictions']), allow_pickle=False) as pred,
                  np.load(old.checked(initial['choices']), allow_pickle=False) as anchor):
                np.testing.assert_array_equal(h['ids'], pred['ids'])
                np.testing.assert_array_equal(h['ids'], anchor['ids'])
                learned = np.stack([pred[k] for k in study.model.MODES], axis=1)
                choice = anchor['anchor']
            for rec in np.unique(h['recording']):
                at = h['recording'] == rec
                q, history, y, v = (h[k][at] for k in ('query', 'history', 'target', 'valid'))
                a = choice[at]
                active = np.flatnonzero(np.isin(q, q[a >= 0]))
                before = target[j, int(rec)]
                if before['edits'] != int((a >= 0).sum()):
                    raise ValueError('Prescribed intervention count changed')
                cached = old.read(menu.PRIVATE / f'view_{j}_{rec}.json')
                incumbent = cached['expanded']
                if cached['key'] != before['key'] or {k: w for k, w in incumbent.items() if k != 'actions'} != before['expanded']:
                    raise ValueError('Analytic reference receipt mismatch')
                key = dict(registration=old.sha(reg), cell=j, recording=int(rec))
                path = out / f'view_{j}_{rec}.json'
                cv, ols = prior.study.motion.bank(history).swapaxes(0, 1)[:2] if len(active) else (None, None)
                bank = causal_bank(history[active]) if len(active) else np.empty((0, 0, 12, 2))
                width = bank.shape[1]
                if len(active):
                    c0 = cv[active, None]
                    delta = learned[at][active]-c0
                    bank = np.concatenate((bank, *(c0+f*delta for f in FRACTIONS)), axis=1)
                args = (cv, ols, bank, y, v, q, active, a, c['cut'])
                if not path.exists():
                    inherit = incumbent['actions'] if incumbent['status'] == 'verified' else None
                    if inherit is not None and menu.receipt(inherit, *args) != incumbent['receipt']:
                        raise ValueError('Analytic witness does not replay')
                    solved = menu.search(*args, inherited=inherit)
                    old.once(path, old.stable(dict(key=key, solved=solved)))
                z = old.read(path)
                solved = z['solved']
                if (z['key'] != key or (solved['actions'] is not None
                        and menu.receipt(solved['actions'], *args) != solved['receipt'])):
                    raise ValueError('Learned witness replay changed')
                if before['expanded']['status'] == 'verified' and solved['status'] != 'verified':
                    raise ValueError('Lost an analytic witness')
                count = sum(int(arm >= width) for _, arm in solved['actions']) if solved['status'] == 'verified' else 0
                rows.append(dict(key=key, source=before['site'], seed=i['seed'], edits=before['edits'],
                    before=incumbent['status'], after=solved['status'], learned_path_edits=count,
                    solver=solved['solver'], witness=solved['receipt']))
                prior.donor.prior.prior.source.atomic(out/'heartbeat.json', dict(pid=os.getpid(), cell=j, views=len(rows)))
            print(dict(cell=j, views=len(rows)), flush=True)
    if len(rows) != capacity['summary']['views'] or sum(x['edits'] for x in rows) != capacity['summary']['edits']:
        raise ValueError('Capacity roster mismatch')
    transitions = Counter(f'{x["before"]} -> {x["after"]}' for x in rows)
    sites = sorted({x['source'] for x in rows})
    means = np.array([np.mean([100.*(int(x['after']=='verified')-int(x['before']=='verified'))
                               for x in rows if x['source']==s]) for s in sites])
    draws = np.random.default_rng(73107).integers(len(means), size=(3000, len(means)))
    summary = dict(before=dict(Counter(x['before'] for x in rows)), after=dict(Counter(x['after'] for x in rows)),
        transitions=dict(transitions), learned_path_edits=sum(x['learned_path_edits'] for x in rows),
        effect_pp=float(means.mean()), interval95_pp=np.quantile(means[draws].mean(1), [.025, .975]).tolist())
    old.once(study.OUT/'capacity.json', old.stable(dict(registration=old.artifact(reg), summary=summary, views=rows,
        original_2pct=True, outcome_aware_diagnostic_only=True, independent_confirmation=False, a_labels_opened=False)))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
