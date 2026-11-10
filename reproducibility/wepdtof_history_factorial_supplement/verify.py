"""Rebuild the WEPDTOF history contrasts from anonymous clip scalars."""

from hashlib import sha256
import json
from math import fsum
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'clips.json', 'protocol.md', 'prior_evidence.zip'}


def replay(panel):
    clips, draws = panel['clips'], panel['draws']
    groups = {r['source'] for r in clips}
    if (len(clips) != 15 or groups != set(range(13)) or sum(r['stable_histories'] for r in clips) != 353
            or sum(r['roster'] for r in clips) != 431 or len(draws) != 3000
            or any(len(d) != 13 or any(type(i) is not int or i not in groups for i in d) for d in draws)):
        raise ValueError('Invalid WEPDTOF source roster')

    def average(x):
        return fsum(x)/len(x)

    def estimate(f):
        group = [average([f(r) for r in clips if r['source'] == i]) for i in range(13)]
        q = quantiles([average([group[i] for i in d]) for d in draws], n=40, method='inclusive')
        return dict(point=average(group), ci95=[q[0], q[-1]])

    def effect(r, m):
        return r[m][0]-r[m][3]

    def anchor(r, m):
        d, p, v, b = r[m]
        return .5*((d-p)+(v-b))

    def velocity(r, m):
        d, p, v, b = r[m]
        return .5*((d-v)+(p-b))

    return dict(cv={k: estimate(lambda r: f(r, 'cv')) for k, f in
                    [('anchor', anchor), ('velocity', velocity), ('total', effect)]},
                ols4={k: estimate(lambda r: f(r, 'ols4')) for k, f in
                      [('anchor', anchor), ('velocity', velocity), ('total', effect)]},
                gap=estimate(lambda r: (r['cv'][0]-r['ols4'][0])-(r['cv'][3]-r['ols4'][3])),
                refit={k: estimate(lambda r: r['refit_gains'][k]) for k in ('ade', 'fde', 'p95')},
                risk_passes=sum(r['risk_pass'] for r in clips))


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
    return dict(inherited=parent, wepdtof_history_factorial=replay(json.loads((root/'clips.json').read_text())))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
