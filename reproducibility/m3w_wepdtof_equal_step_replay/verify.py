"""Replay WEPDTOF equal-step statistics from anonymous source means."""

from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import runpy
from statistics import mean
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'supplement.pdf', 'equal_steps.json', 'prior_evidence.zip'}
CONTROLS = ('fixed_independent', 'fixed_uniform', 'fixed_no_pair')
VARIANTS = ('original', 'smaller_peak', 'matched_steps')
FIELDS = ('observed', 'lower', 'upper', 'relative_observed', 'relative_lower', 'relative_upper')


def replay(data):
    draws = data['draws']
    if (len(draws) != 3000 or any(len(d) != 10 or any(type(i) is not int or not 0 <= i < 10 for i in d) for d in draws)
            or set(data['source_values']) != set(CONTROLS)):
        raise ValueError('Unexpected estimable source roster')

    def check(v, expected):
        if len(v) != expected['sources'] or len(v) != 10:
            raise ValueError('A paired arm changed its source support')
        dist = sorted(mean(v[i] for i in d) for d in draws)

        def q(p):
            i, f = divmod(p*(len(dist)-1), 1)
            i = int(i)
            return dist[i]+f*(dist[i+1]-dist[i])

        if not all(isclose(a, b, rel_tol=0., abs_tol=1e-11) for a, b in zip(
                (mean(v), q(.025), q(.975)), (expected['point'], *expected['ci95']), strict=True)):
            raise ValueError('Bootstrap statistics do not reproduce')

    for arm in CONTROLS:
        counts = data['counts'][arm]
        if (counts['eligible_queries'], counts['collapsed_queries'], counts['retained_edits']) != (198, 0, 366):
            raise ValueError('Counts do not match the frozen WEPDTOF budget')
        for field in FIELDS:
            z = [data['source_values'][arm][v][field] for v in VARIANTS]
            if any(len(v) != 13 for v in z) or len({tuple(x is None for x in v) for v in z}) != 1:
                raise ValueError('Variant source masks differ')
            source = [[x for x in v if x is not None] for v in z]
            for v, arr in zip(VARIANTS, source, strict=True):
                check(arr, data['by_comparator'][arm][v][field])
            check([a-b for a, b in zip(source[2], source[0], strict=True)],
                  data['equal_steps_minus_original'][arm][field])
            if arm == 'fixed_no_pair' and any(x != 0. for arr in source for x in arr):
                raise ValueError('Identical joint and additive forecasts must have zero contrast')
    if data['counts']['fixed_no_pair']['identical_edited_views'] != 36:
        raise ValueError('The observed additive identity count differs')
    r = data['by_comparator']['fixed_independent']['matched_steps']
    return dict(edited_queries=198, edits_per_arm=366, sources=10, additive_identity_views=36,
                observed_pp=r['observed']['point'], envelope_pp=[r['lower']['point'], r['upper']['point']])


def verify(root):
    root = Path(root)
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected reviewer archive members')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed reviewer archive member')
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for name in names:
            (folder/name).write_bytes(z.read(name))
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    data = json.loads((root/'equal_steps.json').read_text())
    return dict(inherited=parent, wepdtof_equal_steps=replay(data))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
