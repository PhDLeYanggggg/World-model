"""Recompute the query selection contrast without source trajectories."""

from hashlib import sha256
import json
from math import fsum, isfinite
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'query_mask.json', 'protocol.md', 'prior_evidence.zip'}
MASKS = ('archived_mask', 'gain_mask', 'query_mask')


def replay(panel):
    sites, budgets, draws, effects = (panel[k] for k in ('sites', 'budgets', 'draws', 'location_means'))
    if (len(sites) != 7 or len(budgets) != 126 or len(draws) != 3000 or
            any(len(r) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in r) for r in draws)):
        raise ValueError('Invalid location bootstrap or roster')
    if (any(set(r) != set(sites[0]) for r in sites)
            or any(type(v) is not int or v < 0 for r in sites for v in r.values())):
        raise ValueError('Invalid site counts')
    counts = {k: sum(r[k] for r in sites) for k in sites[0]}
    for n, *v in budgets:
        if (type(n) is not int or n < 0 or len(v) != 3 or not all(isfinite(x) and x >= 0. for x in v)
                or (n == 0) != (v[0] == 0.) or any(x < .95*v[0]-1e-10 or x > v[0]+1e-10 for x in v)):
            raise ValueError('Budget range or count changed')
    if (sum(r[0] for r in budgets) != counts['swaps']
            or any(not 0 <= counts[k+'_qualifying'] <= counts[k+'_opportunities'] <= counts['swaps'] for k in MASKS)
            or not max(counts[k+'_qualifying'] for k in MASKS) <= counts['verified_hindsight_qualifying'] <= counts['swaps']):
        raise ValueError('Count or oracle mismatch')

    def interval(v):
        if len(v) != 7 or not all(isfinite(x) for x in v):
            raise ValueError('Invalid location means')
        q = quantiles([fsum(v[i] for i in ids)/7 for ids in draws], n=40, method='inclusive')
        return dict(point=fsum(v)/7, ci95=[q[0], q[-1]])

    capture = {m: interval([100.*(r['query_mask_'+m]-r['archived_mask_'+m])/r['swaps'] for r in sites])
               for m in ('opportunities', 'qualifying')}
    return dict(counts=counts, spend={k: fsum(r[i+1] for r in budgets) for i, k in enumerate(MASKS)},
                paired={k: {m: interval(v) for m, v in c.items()} for k, c in effects.items()},
                query_minus_archived_count_rate_pp=capture)


def verify(root):
    root = Path(root)
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected review archive members')
    for k, v in hashes.items():
        p = root/k
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != v:
            raise ValueError('Changed review input')
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        sub = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (sub/n).write_bytes(z.read(n))
        previous = runpy.run_path(str(sub/'verify.py'))['verify'](sub)
    result = replay(json.loads((root/'query_mask.json').read_text()))
    c, p = result['counts'], previous['recipient_eligibility']['counts']
    if (c['swaps'], c['views'], c['archived_mask_opportunities'], c['archived_mask_qualifying']) != (
            p['gain_swaps'], p['views'], p['opportunity_on_mask'], p['gain_strict']):
        raise ValueError('The frozen previous mask did not reproduce')
    return dict(inherited=previous, query_mask=result)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
