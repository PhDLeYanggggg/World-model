"""Replay equal-replacement-count results from anonymous location summaries."""

from hashlib import sha256
from io import BytesIO
import json
from math import fsum, isfinite
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'matched.json', 'protocol.md', 'prior_evidence.zip'}


def replay(panel):
    draws = panel['draws']
    loc = panel['location_counts']
    if (len(loc) != 7 or len(draws) != 3000
            or any(len(x) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in x) for x in draws)):
        raise ValueError('Invalid grouped data or bootstrap')
    totals, effects = {}, {}
    for q in ('ego_count', 'fixed_15pct'):
        group = [r[q] for r in loc]
        if any(g.keys() != group[0].keys() or any(not isfinite(v) or v < 0 for v in g.values()) for g in group):
            raise ValueError('Invalid count rows')
        for g in group:
            n = g['matched_swaps']
            if not (max(g['interval_qualified'], g['direct_qualified']) <= g['hindsight_qualified'] <= n
                    <= min(g['interval_original_swaps'], g['direct_original_swaps'], g['rotated'])
                    and g['same_recording_and_angle'] <= g['same_recordings'] <= n
                    and max(g['interval_random_qualified'], g['direct_random_qualified']) <= n):
                raise ValueError('Invalid equal-count summary')
        totals[q] = {k: fsum(x[k] for x in group) for k in group[0]}
        effects[q] = {}
        for key, values in panel['site_means'][q].items():
            if len(values) != 7 or not all(isfinite(v) for v in values):
                raise ValueError('Invalid paired effects')
            cut = quantiles([fsum(values[i] for i in row)/7 for row in draws], n=40, method='inclusive')
            effects[q][key] = dict(point=fsum(values)/7, ci95=[cut[0], cut[-1]])
    return dict(totals=totals, effects=effects)


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected replay roster')
    for name, expected in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError('Changed replay member')
    with TemporaryDirectory() as tmp:
        sub = Path(tmp)
        with ZipFile(BytesIO((root/'prior_evidence.zip').read_bytes())) as z:
            names = z.namelist()
            if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
                raise ValueError('Invalid inherited filename')
            for n in names:
                (sub/n).write_bytes(z.read(n))
        inherited = runpy.run_path(str(sub/'verify.py'))['verify'](sub)
    return dict(inherited=inherited, equal_replacements=replay(json.loads((root/'matched.json').read_text())))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
