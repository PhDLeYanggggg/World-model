"""Replay the policy-recipient ablation and the previous review evidence."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory


OLD = {'README.md', 'frontier.json', 'decision_evidence.zip', 'ranking.json', 'verify_frontier.py',
       'verify_exchange.py', 'context.json', 'verify_context.py', 'angular.json', 'verify_angular.py',
       'vehicle.json', 'verify_vehicle.py', 'vehicle_context.json', 'verify_context_v11.py',
       'vehicle_action.json', 'verify_action_v12.py', 'vehicle_screen.json', 'verify_screen_v13.py',
       'vehicle_nominee.json', 'verify_nominee_v14.py', 'vehicle_family.json'}
MEMBERS = OLD | {'verify.py', 'verify_family_v15.py', 'vehicle_policy_anchor.json'}


def replay(data):
    groups, draws = data['locations'], data['draws']
    if len(groups) != 7 or len(draws) != 3000 or any(len(x) != 7 or any(
            not isinstance(i, int) or not 0 <= i < 7 for i in x) for x in draws):
        raise ValueError('Invalid anonymous bootstrap')
    counts, effects = groups[0]['counts'].keys(), groups[0]['effects'].keys()
    arms = ('margin', 'random_pointwise', 'policy_pointwise', 'random_budget', 'policy_budget')
    for r in groups:
        c, e = r['counts'], r['effects']
        if (c.keys() != counts or e.keys() != effects or any(not isinstance(v, int) or v < 0 for v in c.values())
                or any(not isfinite(v) for v in e.values()) or not 0 <= c['successful_nominees'] <= c['views']):
            raise ValueError('Invalid location values')
        for q in ('ego_count', 'fixed_15pct'):
            if len({c[f'{a}.{q}.count'] for a in arms}) != 1:
                raise ValueError('Lost matched budget')
    totals = {k: sum(r['counts'][k] for r in groups) for k in counts}
    if (totals['views'], totals['successful_nominees']) != (8874, 420):
        raise ValueError('Lost original nominee roster')
    k = (1, 3, 6, 24)
    hits = [totals[f'shortlist.hit_{i}'] for i in k]
    caps = [totals[f'shortlist.ego_count_cap_{i}'] for i in k]
    if hits != sorted(hits) or caps != sorted(caps) or (hits[0], hits[-1], caps[0]) != (420, 1393, 238):
        raise ValueError('Changed shortlist capacity')
    intervals = {}
    for k in effects:
        m = [r['effects'][k] for r in groups]
        bounds = quantiles([sum(m[i] for i in x)/7 for x in draws], n=40, method='inclusive')
        intervals[k] = dict(point=sum(m)/7, ci95=[bounds[0], bounds[-1]])
    return dict(counts=totals, effects=intervals)


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected recipient comparison archive')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed archive member')
    with TemporaryDirectory() as directory:
        sub = Path(directory)
        for name in OLD:
            (sub/name).write_bytes((root/name).read_bytes())
        (sub/'verify.py').write_bytes((root/'verify_family_v15.py').read_bytes())
        (sub/'manifest.json').write_text(json.dumps({p.name: sha256(p.read_bytes()).hexdigest() for p in sub.iterdir()}))
        old = runpy.run_path(str(sub/'verify.py'))['verify'](sub)
    return dict(inherited=old, vehicle_policy_anchor=replay(json.loads((root/'vehicle_policy_anchor.json').read_text())))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
