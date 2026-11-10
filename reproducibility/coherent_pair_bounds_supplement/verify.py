"""Replay the shared-position bounds on anonymous site summaries."""

from hashlib import sha256
from io import BytesIO
import json
from math import fsum, isclose, isfinite
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'method.md', 'verify.py', 'coherent_pairs.json', 'prior_evidence.zip'}


def replay(d, base):
    draws = d['draws']
    if len(draws) != 3000 or any(len(r) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in r) for r in draws):
        raise ValueError('Invalid location resampling')
    effects = {}
    for q in ('ego_count', 'fixed_15pct'):
        effects[q] = {}
        for arm in ('anchor_difference', 'pointwise', 'incumbent'):
            c = d['counts'][q][arm]
            if (not 0 <= c.get('both_signs_witnessed', 0) <= c.get('new_straddles_zero', 0)
                    <= c.get('old_straddles_zero', 0) <= c.get('different_views', 0)
                    or sum(c.get(k, 0) for k in ('single_anchor_stars', 'near_translation_stars',
                                               'other_multianchor_stars')) != c.get('informative_stars', 0)):
                raise ValueError('Invalid witness or star count')
            row = {}
            for k, v in d['site_means'][q][arm].items():
                if len(v) != 7 or not all(isfinite(x) for x in v):
                    raise ValueError('Invalid anonymous location means')
                cuts = quantiles([fsum(v[i] for i in r)/7 for r in draws], n=40, method='inclusive')
                row[k] = dict(point=fsum(v)/7, ci95=[cuts[0], cuts[-1]])
            for new, old in (('old_lower', 'lower'), ('old_upper', 'upper'), ('observed', 'observed')):
                if not isclose(row[new]['point'], base[q][arm][old]['point'], abs_tol=1e-12):
                    raise ValueError('Changed original point estimate')
            old_width = row['old_upper']['point']-row['old_lower']['point']
            new_width = row['new_upper']['point']-row['new_lower']['point']
            if not -1e-12 <= new_width <= old_width+1e-12:
                raise ValueError('Coherent bound became wider')
            effects[q][arm] = row
    return dict(counts=d['counts'], effects=effects)


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected archive members')
    for name, expected in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError('Changed evidence member')
    with TemporaryDirectory() as tmp:
        sub = Path(tmp)
        with ZipFile(BytesIO((root/'prior_evidence.zip').read_bytes())) as z:
            names = z.namelist()
            if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
                raise ValueError('Invalid inherited filename')
            for n in names:
                (sub/n).write_bytes(z.read(n))
        inherited = runpy.run_path(str(sub/'verify.py'))['verify'](sub)
    base = inherited['inherited']['pair_angle']['missing_label_bounds']
    return dict(inherited=inherited, coherent_pairs=replay(json.loads((root/'coherent_pairs.json').read_text()), base))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
