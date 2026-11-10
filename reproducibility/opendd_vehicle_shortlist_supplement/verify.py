"""Verify the anonymous shortlist panel and replay inherited panels."""

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
       'vehicle_nominee.json', 'verify_nominee_v14.py', 'vehicle_family.json', 'verify_family_v15.py',
       'vehicle_policy_anchor.json'}
MEMBERS = OLD | {'verify.py', 'verify_policy_v16.py', 'vehicle_shortlist.json'}
ARMS = ('margin', 'ols_fixed', 'strict_fixed', 'utility_fixed', 'strict_top1', 'utility_top1',
        'strict_top3', 'utility_top3')


def replay(data):
    groups, draws = data['locations'], data['draws']
    if len(groups) != 7 or len(draws) != 3000 or any(
        len(x) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in x) for x in draws
    ):
        raise ValueError('Invalid anonymous bootstrap')
    counts, effects = groups[0]['counts'].keys(), groups[0]['effects'].keys()
    for row in groups:
        c, e = row['counts'], row['effects']
        if (c.keys() != counts or e.keys() != effects or any(type(v) is not int or v < 0 for v in c.values())
                or any(not isfinite(v) for v in e.values())
                or not 0 <= c['strict_top1'] <= c['strict_top3'] <= c['views']):
            raise ValueError('Invalid location values')
        for q in ('ego_count', 'fixed_15pct'):
            if len({c[f'{a}.{q}.count'] for a in ARMS}) != 1:
                raise ValueError('Lost matched budget')
    totals = {k: sum(r['counts'][k] for r in groups) for k in counts}
    if (totals['views'], totals['strict_top1'], totals['strict_top3']) != (8874, 420, 802):
        raise ValueError('Changed nominee roster')
    cap = data['fixed_mask_capacity']
    for q, count in (('ego_count', 526), ('fixed_15pct', 1368)):
        c = cap[q]
        if (c['count'] != count or totals[f'margin.{q}.count'] != count
                or c['strict_top1'] != totals[f'margin.{q}.strict']
                or not c['strict_top1'] <= c['strict_top3'] <= min(count, totals['strict_top3'])
                or any(totals[f'{a}.{q}.strict'] > c['strict_top3']
                       for a in ('ols_fixed', 'strict_fixed', 'utility_fixed'))):
            raise ValueError('Inconsistent fixed-mask capacity')
    intervals = {}
    for k in effects:
        means = [r['effects'][k] for r in groups]
        bounds = quantiles([sum(means[i] for i in x)/7 for x in draws], n=40, method='inclusive')
        intervals[k] = dict(point=sum(means)/7, ci95=[bounds[0], bounds[-1]])
    return dict(counts=totals, effects=intervals, fixed_mask_capacity=cap)


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected shortlist archive')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed archive member')
    with TemporaryDirectory() as directory:
        sub = Path(directory)
        for name in OLD:
            (sub/name).write_bytes((root/name).read_bytes())
        (sub/'verify.py').write_bytes((root/'verify_policy_v16.py').read_bytes())
        (sub/'manifest.json').write_text(json.dumps({p.name: sha256(p.read_bytes()).hexdigest() for p in sub.iterdir()}))
        old = runpy.run_path(str(sub/'verify.py'))['verify'](sub)
    return dict(inherited=old, vehicle_shortlist=replay(json.loads((root/'vehicle_shortlist.json').read_text())))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
