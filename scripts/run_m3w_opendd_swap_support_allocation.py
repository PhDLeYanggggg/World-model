"""Check whether improved OpenDD ranking survives the original budget."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_swap_support as ranks

old, np, base = ranks.old, ranks.np, ranks.base
read, write, sha = ranks.read, ranks.write, ranks.sha
OUT = ranks.OUT.parent/'opendd_swap_support_allocation_v1'
LOCAL = ranks.LOCAL.parent/OUT.name


def register():
    parent, source, original, _, previous, _ = ranks.register()
    paths = (ranks.OUT/'freeze.json', ranks.OUT/'readout.json')
    f, r = map(read, paths)
    if (f['key'] != parent or r['freeze'] != sha(paths[0]) or not r['summary']['allocation_screen']
            or not r['original_2pct'] or r['external_outcomes_read'] or r['new_allocation_executed']):
        raise ValueError('Require a passing, matched ranking screen')
    dest = OUT/'registration.json'
    write(dest, dict(files={str(p.relative_to(base.prior.ROOT)): sha(p) for p in
        (*paths, old.OUT/'readout.json', Path(__file__), OUT/'protocol.md')},
        relative_weight=.5, original_2pct=True, external_outcomes_read=False), immutable=True)
    return sha(dest), source, original, previous, f


def nominate(prev, scores):
    action, cost, keep = (prev[k] for k in ('actions', 'cost', 'budget_anchor'))
    legal = action >= 0
    prob = scores['footprint'][:, :, 3]
    if prob.shape != (int(legal.sum()), 2) or not np.isfinite(prob).all():
        raise ValueError('Rank scores do not bind the old nominations')
    x = np.zeros((len(action), 2))
    x[legal] = prob
    a, s = old.prior.model.allocate(x[:, 0], cost, keep)
    b, t = old.prior.model.allocate(x[:, 0]+.5*x[:, 1], cost, keep)
    out = {k: prev[k] for k in ('actions', 'cost', 'budget_anchor', 'opportunity')}
    return dict(out, trajectory=a, joint=b), dict(trajectory=s, joint=t)


def freeze(key, original, previous, ranks_freeze):
    rows = {}
    for name, c in original['cells'].items():
        old.old.check_cell(c)
        for held, orig in c['packets'].items():
            tag = f"{held}_{c['seed']}"
            a, b = previous['cells'][tag]['packets'][name], ranks_freeze['cells'][tag]['packets'][name]
            if b['parent'] != a['sha256'] or held in ranks_freeze['cells'][tag]['fit']:
                raise ValueError('Wrong nomination or held model')
            p = ranks.checked_local(a, old.LOCAL, orig['sha256'])
            q = ranks.checked_local(b, ranks.LOCAL, orig['sha256'])
            f = LOCAL/f'{tag}_{name}.npz'
            with np.load(p, allow_pickle=False) as prev, np.load(q, allow_pickle=False) as pred:
                decisions, status = nominate(prev, pred)
            rows[f'{tag}/{name}'] = dict(file=f.name, sha256=base.parent.indices.save_npz(f, decisions),
                original=orig['sha256'], rank=b['sha256'], status=status)
    write(OUT/'freeze.json', dict(key=key, rows=rows, external_outcomes_read=False), immutable=True)


def score(key, source, original):
    path = OUT/'freeze.json'
    old.old.require_published(path)
    f = read(path)
    if f['key'] != key or f['external_outcomes_read'] or len(f['rows']) != 126:
        raise ValueError('Require all frozen allocations')
    features, data = old.old.prior.load_features(source), base.parent.load(source)
    rows, solver = [], Counter()
    for name, cell in original['cells'].items():
        for held, orig in cell['packets'].items():
            seed = cell['seed']
            r = f['rows'][f'{held}_{seed}/{name}']
            decision = ranks.checked_local(r, LOCAL, orig['sha256'])
            solver.update(f'{k}_{v}' for k, v in r['status'].items())
            p = LOCAL/f'{name}_{held}_score.json'
            identity = dict(key=key, held=held, seed=seed, original=orig['sha256'], action=r['sha256'])
            if not p.exists():
                with ranks.packet(orig) as z, np.load(decision, allow_pickle=False) as a:
                    v = old.evaluate(z, features[held], data[held], cell['cut'], a)
                write(p, dict(identity=identity, **v), immutable=True)
            v = read(p)
            if v['identity'] != identity:
                raise ValueError('Changed score receipt')
            rows.append(v)
        write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase='score', completed=len(rows), total=126))
        if len(rows) % 14 == 0:
            print(f'{len(rows)}/126 directions', flush=True)
    s, b = old.summarize(rows), read(old.OUT/'readout.json')['summary']
    if any(s['counts'][k] != b['counts'][k] for k in ('views', 'swaps', 'opportunity_strict')) or any(
            s['arms'][a][k] != b['arms'][a][k] for a in ('original', 'opportunity') for k in ('risk', 'dual', 'missing')):
        raise ValueError('Archived allocation did not reproduce')
    write(OUT/'readout.json', dict(key=key, freeze=sha(path), summary=s, solver=dict(solver),
        original_2pct=True, external_outcomes_read=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False), immutable=True)
    print(s['counts'], dict(solver), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze', 'score'))
    args = p.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, source, original, previous, scores = register()
        if args.phase == 'freeze':
            freeze(key, original, previous, scores)
        else:
            score(key, source, original)
