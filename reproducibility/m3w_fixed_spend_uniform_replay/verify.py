"""Replay the fixed-spend uniform control from anonymous aggregates."""

from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import runpy
from statistics import mean
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'supplement.pdf', 'fixed_spend_uniform.json', 'prior_evidence.zip'}


def percentile(a, p):
    i, f = divmod(p*(len(a)-1), 1)
    i = int(i)
    return a[i] + f*(a[i+1]-a[i])


def close(a, b):
    return isclose(a, b, rel_tol=0., abs_tol=1e-10)


def replay(data):
    groups, draws, summary = (data[k] for k in ('locations', 'draws', 'summary'))
    if (len(groups) != 7 or any({r['seed'] for r in g} != {17, 29, 43} or len(g) != 3 for g in groups)
            or len(draws) != 3000 or any(len(d) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in d) for d in draws)):
        raise ValueError('Unexpected bootstrap roster')
    totals = [sum(r[k] for g in groups for r in g) for k in ('views', 'frozen', 'hindsight', 'uniform_low', 'uniform_high')]
    if (totals[:3] != [summary[k] for k in ('views', 'frozen_strict', 'hindsight')] or totals[:3] != [606, 55, 210]
            or summary['no_accepted_draws'] or not close(totals[3], totals[4]) or not close(totals[3], summary['expected_passes_low'])):
        raise ValueError('Fixed-spend totals disagree')
    if any(not 0 <= r['uniform_low'] <= r['hindsight'] <= r['views'] for g in groups for r in g):
        raise ValueError('Uniform expectation exceeds hindsight capacity')
    for arm in ('frozen', 'hindsight'):
        rates = [mean(100*(r[arm]-r['uniform_low'])/r['views'] for r in g) for g in groups]
        sample = sorted(mean(rates[i] for i in d) for d in draws)
        ref = data['contrasts'][arm+'_minus_uniform']
        found = [mean(rates), mean(rates), percentile(sample, .025), percentile(sample, .975)]
        if not all(close(a, b) for a, b in zip(found, ref['point_bounds_pp']+ref['location_ci95_pp'], strict=True)):
            raise ValueError('Bootstrap differs from archived results')
    return summary


def verify(root):
    root = Path(root)
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected supplement members')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed supplement file')
    result = replay(json.loads((root/'fixed_spend_uniform.json').read_text()))
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for name in names:
            (folder/name).write_bytes(z.read(name))
        if (root/'supplement.pdf').read_bytes() != (folder/'supplement.pdf').read_bytes():
            raise ValueError('Reader PDF differs from inherited evidence')
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    counts = parent['opendd_full_grid_count_motion']
    if (counts['blind_full_grid'], counts['full_grid']) != (result['frozen_strict'], result['hindsight']):
        raise ValueError('Archived comparators changed')
    return dict(inherited=parent, opendd_fixed_spend_uniform=result)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
