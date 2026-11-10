"""Reproduce the paired support-ranking and fixed-budget allocation evidence."""

from hashlib import sha256
import json
from math import fsum, isclose, isfinite
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'support.json', 'ranking_protocol.md', 'allocation_protocol.md', 'prior_evidence.zip'}


def interval(v, draws):
    if len(v) != 7 or not all(isfinite(x) for x in v):
        raise ValueError('Invalid location estimates')
    q = quantiles([fsum(v[i] for i in d)/7 for d in draws], n=40, method='inclusive')
    return dict(point=fsum(v)/7, ci95=[q[0], q[-1]])


def replay(panel):
    rank, allocation = (panel[k] for k in ('rank_draws', 'allocation_draws'))
    if any(len(x) != 3000 or any(len(d) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in d)
                                for d in x) for x in (rank, allocation)):
        raise ValueError('Invalid location resampling draws')
    r = {k: interval(v, rank) for k, v in panel['rank_effects'].items()}
    a = {k: {m: interval(v, allocation) for m, v in c.items()} for k, c in panel['allocation_effects'].items()}
    gate = (r['relative_footprint_sign_minus_v196_fused']['ci95'][0] > 0.
            and r['trajectory_footprint_sign_minus_v194_gain']['point'] >= 0.)
    counts = panel['allocation_counts']
    if any(type(v) is not int or v < 0 for v in counts.values()) or any(
            counts[k] > counts['swaps'] for k in ('opportunity_strict', 'trajectory_strict', 'joint_strict')):
        raise ValueError('Invalid fixed-budget counts')
    return dict(rank_effects=r, allocation_effects=a, allocation_counts=counts, allocation_screen=gate)


def verify(root):
    root = Path(root)
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected review members')
    for k, v in hashes.items():
        p = root/k
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != v:
            raise ValueError('Changed review inputs')
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    d = replay(json.loads((root/'support.json').read_text()))
    a, b = d['allocation_counts'], parent['realized_swap_value']['counts']
    if any(a[k] != b[k] for k in ('views', 'swaps', 'opportunity_strict')):
        raise ValueError('The original nomination and quota did not replay')
    original = parent['realized_swap_value']['paired']['opportunity_minus_original']
    if any(not isclose(v['point'], original[k]['point'], rel_tol=0., abs_tol=1e-12)
           for k, v in d['allocation_effects']['opportunity_minus_original'].items()):
        raise ValueError('Opportunity comparison changed')
    return dict(inherited=parent, swap_support=d)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
