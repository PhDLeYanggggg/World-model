"""Reproduce the matched public baseline from location-by-seed scalars."""

from hashlib import sha256
import json
from math import fsum, isfinite
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ARMS = ('social_peers', 'social_ego', 'completion', 'cv', 'ols4')
KEYS = ('known_ade_native', 'fde_available_native', 'easy_gain_percent', 'p95_ade_native',
        'cv_gain_lower_pp', 'ols_gain_lower_pp')
CONTRASTS = ('social_peers_minus_ego', 'social_peers_minus_incumbent', 'social_ego_minus_incumbent')
BACKBONES = ('social_ego_minus_eqmotion_ego', 'social_peers_minus_eqmotion_peers')
MEMBERS = {'README.md', 'verify.py', 'opendd_cells.json', 'protocol.md', 'forecast_controls.pdf', 'prior_evidence.zip'}


def replay(panel):
    cells, draws = panel['cells'], panel['draws']
    if (len(cells) != 21 or {(v['site'], v['seed']) for v in cells} != {(i, s) for i in range(7) for s in range(3)}
            or len(draws) != 3000 or any(len(x) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in x)
                                         for x in draws)):
        raise ValueError('Invalid leave-location-out roster')
    if any(set(r['arms']) != set(ARMS) or set(r['paired']) != set(CONTRASTS) or set(r['backbones']) != set(BACKBONES) or
           any(set(v) != set(KEYS) | {'risk_passes', 'recordings'} or
               not 0 <= v['risk_passes'] <= v['recordings'] for v in r['arms'].values()) for r in cells):
        raise ValueError('Unexpected matched-control statistics')

    def estimate(fun):
        sites = [fsum(fun(r) for r in cells if r['site'] == i)/3 for i in range(7)]
        if not all(isfinite(v) for v in sites):
            raise ValueError('Nonfinite site statistic')
        q = quantiles([fsum(sites[i] for i in d)/7 for d in draws], n=40, method='inclusive')
        return dict(mean=fsum(sites)/7, ci95=[q[0], q[-1]], sites=sites)

    metrics = {a: {k: estimate(lambda r: r['arms'][a][k]) for k in KEYS} for a in ARMS}
    for a in ARMS:
        metrics[a]['risk_passes'] = sum(r['arms'][a]['risk_passes'] for r in cells)
    paired = {a: {k: estimate(lambda r: r['paired'][a][j]) for j, k in enumerate(('observed', 'lower'))}
              for a in CONTRASTS}
    backbone = {}
    for a in BACKBONES:
        backbone[a] = {}
        for k in ('observed', 'lower'):
            m = estimate(lambda r: r['backbones'][a][k])
            backbone[a][k] = dict(point=m['mean'], ci95=m['ci95'], sites=m['sites'])
    return dict(metrics=metrics, paired=paired, backbone=backbone)


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
    return dict(inherited=parent, opendd_vehicle_stgcnn=replay(json.loads((root/'opendd_cells.json').read_text())))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
