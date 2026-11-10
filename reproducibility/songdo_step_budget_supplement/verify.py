"""Replay the Songdo time-profile sensitivity from anonymous source means."""

from hashlib import sha256
import json
from math import fsum, isfinite
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'profiles.json', 'protocol.md', 'prior_evidence.zip'}
ARMS = ('original', 'smaller_peak', 'matched_steps')
METRICS = ('known_pp', 'lower_pp', 'upper_pp', 'relative_available_native', 'relative_lower_native')


def replay(panel):
    draws, means = panel['draws'], panel['source_means']
    if (set(means) != set(ARMS) or len(draws) != 3000 or panel['flights'] != 160
            or any(len(row) != 14 or any(type(i) is not int or not 0 <= i < 14 for i in row) for row in draws)):
        raise ValueError('Invalid Songdo source roster')

    def interval(values):
        if len(values) != 14:
            raise ValueError('Require the complete original 14-component roster')
        if any(v is None for v in values):
            return dict(point=None, ci95=None)
        if not all(isfinite(x) for x in values):
            raise ValueError('Invalid component values')
        q = quantiles([fsum(values[i] for i in row)/14 for row in draws], n=40, method='inclusive')
        return dict(point=fsum(values)/14, ci95=[q[0], q[-1]])

    for a in ARMS:
        if set(means[a]) != set(METRICS) or not 0 <= panel['views'][a] <= 480:
            raise ValueError('Invalid matched arm support')
        lo, mid, hi = (means[a][k] for k in ('lower_pp', 'known_pp', 'upper_pp'))
        if any(not x <= y <= z for x, y, z in zip(lo, mid, hi, strict=True) if None not in (x, y, z)):
            raise ValueError('A completion envelope does not contain the observed gain')
    if panel['views']['original'] != 480 or panel['views']['smaller_peak'] != panel['views']['matched_steps']:
        raise ValueError('Sensitivity variants are not paired')
    direct = {a: {k: interval(v) for k, v in rows.items()} for a, rows in means.items()}
    return dict(direct=direct, retention={k: interval(v) for k, v in panel['retained'].items()},
                matched_views=panel['views']['matched_steps'])


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
    return dict(inherited=parent, songdo_step_budget=replay(json.loads((root/'profiles.json').read_text())))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
