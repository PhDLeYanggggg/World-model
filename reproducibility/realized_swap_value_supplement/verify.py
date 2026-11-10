"""Replay direct observed effects and missing-label bounds by location."""

from hashlib import sha256
import json
from math import fsum, isclose, isfinite
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'realized_value.json', 'protocol.md', 'prior_evidence.zip'}


def replay(panel):
    counts, means, draws = (panel[k] for k in ('counts', 'location_means', 'draws'))
    if (len(draws) != 3000 or any(len(v) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in v) for v in draws)
            or any(type(v) is not int or v < 0 for v in counts.values())
            or any(counts[a+'_strict'] > counts['swaps'] for a in ('opportunity', 'trajectory', 'joint'))):
        raise ValueError('Invalid grouped inputs or counts')

    def interval(v):
        if len(v) != 7 or not all(isfinite(x) for x in v):
            raise ValueError('Invalid source means')
        q = quantiles([fsum(v[i] for i in ids)/7 for ids in draws], n=40, method='inclusive')
        return dict(point=fsum(v)/7, ci95=[q[0], q[-1]])

    for m in means.values():
        for observed, lower, upper in (('known_pp', 'lower_pp', 'upper_pp'),
                                       ('relative_observed', 'relative_lower', 'relative_upper')):
            if any(l > o+1e-12 or o > u+1e-12 or not isclose(l+u, 2.*o, abs_tol=1e-12, rel_tol=0.)
                   for o, l, u in zip(m[observed], m[lower], m[upper], strict=True)):
                raise ValueError('Observed and missing-label contrasts do not agree')
    return dict(counts=counts, paired={a: {k: interval(v) for k, v in m.items()} for a, m in means.items()})


def verify(root):
    root = Path(root)
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected replay members')
    for k, v in hashes.items():
        p = root/k
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != v:
            raise ValueError('Changed replay input')
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    d = replay(json.loads((root/'realized_value.json').read_text()))
    old = parent['query_mask']
    a, b = d['counts'], old['counts']
    if (a['views'], a['swaps'], a['opportunity_strict']) != (b['views'], b['swaps'], b['query_mask_qualifying']):
        raise ValueError('Archived opportunity allocation changed')
    for k in ('known_pp', 'lower_pp', 'relative_lower', 'dual_pp'):
        if not isclose(d['paired']['opportunity_minus_original'][k]['point'],
                       old['paired']['query_mask_minus_original'][k]['point'], abs_tol=1e-12, rel_tol=0.):
            raise ValueError('Archived opportunity effects changed')
    return dict(inherited=parent, realized_swap_value=d)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
