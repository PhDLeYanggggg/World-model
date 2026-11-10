"""Recompute the recipient eligibility counts and paired intervals."""

from hashlib import sha256
import json
from math import fsum, isfinite
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'eligibility.json', 'protocol.md', 'prior_evidence.zip'}


def replay(panel):
    rows, draws, effects = (panel[k] for k in ('counts_by_site', 'draws', 'location_means'))
    if (len(rows) != 7 or len(draws) != 3000 or
            any(len(r) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in r) for r in draws)):
        raise ValueError('Use seven location units')
    if (any(set(r) != set(rows[0]) for r in rows)
            or any(type(v) is not int or v < 0 for r in rows for v in r.values())):
        raise ValueError('Invalid site counts')
    n = {k: sum(r[k] for r in rows) for k in rows[0]}
    if (n['archived_swaps'] != n['gain_swaps'] or n['gain_swaps'] != n['eligibility_swaps']
            or not 0 <= n['guarded_swaps'] <= n['gain_swaps'] <= n['views']
            or not 0 <= n['opportunity_on_mask'] <= min(n['opportunity_all'], n['gain_swaps'])
            or any(n[k+'_strict'] > min(n['opportunity_on_mask'], n[k+'_swaps'])
                   for k in ('archived', 'gain', 'eligibility', 'guarded'))):
        raise ValueError('Broken matched-count comparison')
    paired = {}
    for key, metrics in effects.items():
        paired[key] = {}
        for k, v in metrics.items():
            if len(v) != 7 or not all(isfinite(x) for x in v):
                raise ValueError('Invalid location means')
            interval = quantiles([fsum(v[i] for i in r)/7 for r in draws], n=40, method='inclusive')
            paired[key][k] = dict(point=fsum(v)/7, ci95=[interval[0], interval[-1]])
    return dict(counts=n, paired=paired)


def verify(root):
    root = Path(root)
    expected = json.loads((root/'manifest.json').read_text())
    if set(expected) != MEMBERS or {f.name for f in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected archive members')
    for k, v in expected.items():
        p = root/k
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != v:
            raise ValueError('Changed archive input')
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        inherited = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    return dict(inherited=inherited, recipient_eligibility=replay(json.loads((root/'eligibility.json').read_text())))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
