"""Verify anonymous aggregates for relative-trajectory comparisons."""

from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import runpy
from statistics import mean
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'supplement.pdf', 'backbone_relative.json', 'prior_evidence.zip'}


def replay(data):
    groups, draws, support = (data[k] for k in ('locations', 'draws', 'support'))
    if (len(groups) != 7 or any(len(g) != 3 for g in groups) or len(draws) != 3000
            or any(len(d) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in d) for d in draws)):
        raise ValueError('Wrong OpenDD replay roster')
    if any(any(r['support'] != g[0]['support'] for r in g) for g in groups):
        raise ValueError('Query support varies across seeds')
    if any(v != sum(g[0]['support'][k] for g in groups) for k, v in support.items()):
        raise ValueError('Support count differs from archived total')
    if not (support['rows'] == 141783 and support['queries'] == 25596
            and 0 < support['complete_pair_queries'] <= support['observed_queries'] <= support['pair_queries']
            and 0 < support['observed_pair_steps'] <= support['pair_steps']):
        raise ValueError('Incorrect or impossible archived OpenDD support')

    def quantile(x, p):
        i, t = divmod(p*(len(x)-1), 1)
        i = int(i)
        return x[i] + t*(x[i+1]-x[i])

    for public, per_seed in (('absolute', 'absolute'), ('paired', 'contrasts')):
        for arm, fields in data[public].items():
            for field, expected in fields.items():
                x = [mean(r[per_seed][arm][field] for r in g) for g in groups]
                draws_x = sorted(mean(x[i] for i in d) for d in draws)
                found = [*x, mean(x), quantile(draws_x, .025), quantile(draws_x, .975)]
                wanted = expected['sites']+[expected['point'], *expected['ci95']]
                if not all(isclose(a, b, rel_tol=0., abs_tol=1e-10) for a, b in zip(found, wanted, strict=True)):
                    raise ValueError('OpenDD paired interval differs')
    return dict(support=support, absolute=data['absolute'], paired=data['paired'])


def verify(root):
    root = Path(root)
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected replay members')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Replay member changed')
    current = replay(json.loads((root/'backbone_relative.json').read_text()))
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for name in names:
            (folder/name).write_bytes(z.read(name))
        if (root/'supplement.pdf').read_bytes() != (folder/'supplement.pdf').read_bytes():
            raise ValueError('Reader PDF and inherited evidence differ')
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    return dict(inherited=parent, opendd_backbone_relative=current)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
