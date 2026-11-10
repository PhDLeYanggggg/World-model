"""Replay angle agreement and the inherited aggregate panels."""

from hashlib import sha256
from io import BytesIO
import json
from math import isfinite
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory
from zipfile import ZipFile


MEMBERS = {'README.md', 'verify.py', 'angle_agreement.json', 'prior_evidence.zip'}


def replay(d):
    groups, draws = d['locations'], d['draws']
    if len(groups) != 7 or len(draws) != 3000 or any(len(x) != 7 or any(
            type(i) is not int or not 0 <= i < 7 for i in x) for x in draws):
        raise ValueError('Invalid location bootstrap')
    counts, effects = groups[0]['counts'].keys(), groups[0]['effects'].keys()
    for group in groups:
        c, e = group['counts'], group['effects']
        if (c.keys() != counts or e.keys() != effects or any(type(n) is not int or n < 0 for n in c.values())
                or not all(isfinite(v) for v in e.values())):
            raise ValueError('Invalid anonymous readout')
    total = {k: sum(r['counts'][k] for r in groups) for k in counts}
    for q, n in (('ego_count', 526), ('fixed_15pct', 1368)):
        for a in ('margin', 'utility', 'trajectory', 'agreement', 'calibrated_agreement'):
            root = f'{q}.{a}'
            if (total[f'{root}.rotated'] != n or not total[f'{root}.qualified_joint_gain']
                    <= min(total[f'{q}.capacity.qualified'], total[f'{root}.swaps']) <= n):
                raise ValueError('Lost matched mask or hindsight ordering')
    intervals = {}
    for k in effects:
        values = [r['effects'][k] for r in groups]
        cuts = quantiles([sum(values[i] for i in x)/7 for x in draws], n=40, method='inclusive')
        intervals[k] = dict(point=sum(values)/7, ci95=[cuts[0], cuts[-1]])
    return dict(counts=total, effects=intervals)


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected angle agreement archive')
    for name, expected in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError('Changed member')
    with TemporaryDirectory() as folder:
        sub = Path(folder)
        with ZipFile(BytesIO((root/'prior_evidence.zip').read_bytes())) as z:
            names = z.namelist()
            if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
                raise ValueError('Unexpected inherited archive path')
            for name in names:
                (sub/name).write_bytes(z.read(name))
        old = runpy.run_path(str(sub/'verify.py'))['verify'](sub)
    return dict(inherited=old, angle_agreement=replay(json.loads((root/'angle_agreement.json').read_text())))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
