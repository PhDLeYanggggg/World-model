"""Replay fixed-nomination easy-reference experiments from scalar receipts."""

from hashlib import sha256
import json
from math import fsum, isfinite
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'directions.json', 'protocol.md', 'prior_evidence.zip'}
ARMS = ('sign', 'sign_times_reference', 'reference_only')
COUNTS = ('released', 'risk', 'dual', 'strict', 'missing', 'eligible')


def replay(panel):
    rows, draws = panel['directions'], panel['draws']
    roster = {(h, c, s) for h in range(7) for c in range(7) if h != c for s in range(3)}
    if len(rows) != len(roster) or {tuple(x['roles']) for x in rows} != roster:
        raise ValueError('The complete 126 direction roster is required')
    if len(draws) != 3000 or any(len(r) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in r) for r in draws):
        raise ValueError('The 3000 location resamples are required')

    def interval(values):
        tails = quantiles([fsum(values[i] for i in draw)/7 for draw in draws], n=40, method='inclusive')
        return dict(point_pp=fsum(values)/7, ci95_pp=[tails[0], tails[-1]])

    counts, sites = {}, {}
    for a in ARMS:
        x = [r['arms'][a] for r in rows]
        if any(any(type(v[k]) is not int or v[k] < 0 for k in (*COUNTS, 'calibration_count'))
               or not v['strict'] <= v['dual'] <= v['risk'] <= v['released'] <= v['eligible']
               or v['missing'] > v['released']-v['risk']
               or (not v['calibration_count'] and v['released'])
               or not isfinite(v['lower_cv_pp']) for v in x):
            raise ValueError('Inconsistent release receipt')
        counts[a] = {k: sum(v[k] for v in x) for k in COUNTS}
        counts[a]['calibration_active_directions'] = sum(v['calibration_count'] > 0 for v in x)
        sites[a] = [fsum(r['arms'][a]['lower_cv_pp'] for r in rows if r['roles'][0] == h)/18 for h in range(7)]
    if any(len({r['arms'][a]['eligible'] for a in ARMS}) != 1 for r in rows):
        raise ValueError('Arms used different nomination rosters')
    if [counts['sign'][k] for k in COUNTS[:5]] != [32, 12, 8, 6, 16]:
        raise ValueError('The original sign comparator does not replay')
    return dict(counts=counts, cv={a: interval(x) for a, x in sites.items()},
                paired={a: interval([x-y for x, y in zip(sites[a], sites['sign'], strict=True)]) for a in ARMS[1:]})


def verify(root):
    root = Path(root)
    h = json.loads((root/'manifest.json').read_text())
    if set(h) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected supplement contents')
    for k, v in h.items():
        p = root/k
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != v:
            raise ValueError('Changed supplement input')
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    return dict(inherited=parent, easy_reference_release=replay(json.loads((root/'directions.json').read_text())))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
