"""Reproduce the nested calibration roster, release counts and intervals."""

from hashlib import sha256
import json
from math import fsum, isfinite
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'nested_release.json', 'protocol.md', 'prior_evidence.zip'}
ARMS = ('additive', 'sign', 'magnitude')
COUNTS = ('released', 'risk', 'dual', 'strict', 'missing', 'eligible')


def replay(d):
    roster, draws = d['directions'], d['draws']
    expected = {(h, c, s) for h in range(7) for c in range(7) if h != c for s in range(3)}
    if (len(roster) != 126 or {tuple(r['roles']) for r in roster} != expected or len(draws) != 3000
            or any(len(r) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in r) for r in draws)):
        raise ValueError('Invalid disjoint roster or resamples')

    def interval(v):
        if len(v) != 7 or not all(isfinite(x) for x in v):
            raise ValueError('Invalid location means')
        q = quantiles([fsum(v[i] for i in ids)/7 for ids in draws], n=40, method='inclusive')
        return dict(point=fsum(v)/7, ci95=[q[0], q[-1]])

    summary = {}
    for a in ARMS:
        rows = [r['arms'][a] for r in roster]
        if any(set(r) != set(COUNTS) | {'calibration_count'} or any(type(x) is not int or x < 0 for x in r.values())
               or not r['strict'] <= r['dual'] <= r['risk'] <= r['released'] <= r['eligible']
               or r['missing'] > r['released'] or (r['calibration_count'] == 0 and r['released'] != 0) for r in rows):
            raise ValueError('A release is inconsistent with the frozen calibration')
        summary[a] = {k: sum(r[k] for r in rows) for k in COUNTS}
        summary[a].update(calibration_active_directions=sum(r['calibration_count'] > 0 for r in rows),
            held_active_directions=sum(r['released'] > 0 for r in rows),
            active_directions_all_strict=sum(0 < r['released'] == r['strict'] for r in rows),
            effects={k: interval(v) for k, v in d['effects'][a].items()})
    if [summary['additive'][k] for k in ('released', 'risk', 'dual')] != [65, 32, 19]:
        raise ValueError('The previously archived additive release did not replay')
    return summary


def verify(root):
    root = Path(root)
    h = json.loads((root/'manifest.json').read_text())
    if set(h) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected review roster')
    for k, v in h.items():
        p = root/k
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != v:
            raise ValueError('Changed review input')
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    return dict(inherited=parent, nested_swap_release=replay(json.loads((root/'nested_release.json').read_text())))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
