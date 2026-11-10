"""Replay the new equal-training table and the inherited main evidence."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from statistics import mean
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'supplement.pdf', 'listwise.json', 'prior_evidence.zip'}
ARMS = ('incumbent', 'aligned', 'uniform', 'point_original', 'list_original', 'list_full_grid')
FIELDS = ('original', 'full_grid', 'material_full_grid')


def replay(data, quantile):
    arms, sites, draws = (data[k] for k in ('arms', 'sites', 'draws'))
    if (set(arms) != set(ARMS) or len(sites) != 7 or any(len(g) != 3 for g in sites)
            or data['counts'] != {'views': 606, 'edits': 780, 'switched': 101}
            or any(sum(r['counts'][k] for g in sites for r in g) != v for k, v in data['counts'].items())
            or len(draws) != 3000 or any(len(row) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in row) for row in draws)):
        raise ValueError('Unexpected same-mask training and held roster')
    for name, row in arms.items():
        o, f, m = (row[k] for k in FIELDS)
        if (not 0 <= m <= f <= o <= 606
                or any(abs(sum(r['arms'][name][k] for g in sites for r in g)-row[k]) > 1e-10 for k in FIELDS)):
            raise ValueError('Per-location results do not reproduce the table')
    pairs = (('list_original', 'point_original'), ('list_full_grid', 'list_original'),
             ('list_full_grid', 'uniform'), ('list_full_grid', 'incumbent'), ('list_full_grid', 'aligned'))
    if set(data['contrasts']) != {f'{a}_minus_{b}' for a, b in pairs}:
        raise ValueError('Unexpected paired comparison set')
    for a, b in pairs:
        x = [mean(100*(r['arms'][a]['full_grid']-r['arms'][b]['full_grid'])/r['counts']['views'] for r in g) for g in sites]
        z = sorted(mean(x[i] for i in d) for d in draws)
        expected = data['contrasts'][f'{a}_minus_{b}']
        if any(abs(m-n) > 1e-10 for m, n in zip((mean(x), quantile(z, .025), quantile(z, .975)),
                (expected['point_pp'], *expected['ci95_pp']), strict=True)):
            raise ValueError('Paired interval failed replay')
    return {k: arms[k]['full_grid'] for k in ARMS}


def verify(root):
    root = Path(root)
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected supplement members')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed supplement file')
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        if (folder/'supplement.pdf').read_bytes() != (root/'supplement.pdf').read_bytes():
            raise ValueError('Expected the unchanged reader PDF')
        helpers = runpy.run_path(str(folder/'verify.py'))
        prior = helpers['verify'](folder)
        result = replay(json.loads((root/'listwise.json').read_text()), helpers['quantile'])
        for k in ('incumbent', 'aligned', 'uniform'):
            if abs(result[k]-prior['opendd_full_grid_exchange'][k][1]) > 1e-10:
                raise ValueError('Controls do not reproduce the inherited full-grid result')
    return dict(inherited=prior, opendd_listwise_full_grid=result)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
